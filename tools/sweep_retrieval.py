#!/usr/bin/env python3
"""Run named public-22 retrieval experiments on one frozen chunk catalog."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

from compare_retrieval import load_frozen_run
from evaluate_retrieval import evaluate
from prepare_evaluation import ROOT, prepare, write_json, fingerprint

sys.path.insert(0, str(ROOT))
from src.retrieval import BM25Retriever, rank_chunks, select_context, MAX_CONTEXT_TOKENS


def number_label(value):
    return f"{value:g}".replace(".", "p")


def experiment_name(config):
    method = config["method"]
    if method == "dense":
        return "dense-reference"
    bm25 = f"k1_{number_label(config['k1'])}-b{number_label(config['b'])}"
    if method == "bm25":
        return f"bm25-{bm25}"
    return (f"hybrid-dw{number_label(config['dense_weight'])}"
            f"-bw{number_label(config['bm25_weight'])}-{bm25}"
            f"-rrf{number_label(config['rrf_k'])}")


def configurations(k1_values, b_values, bm25_weights, rrf_values):
    """A dense control, lexical controls, then the complete hybrid grid."""
    yield {"method": "dense"}
    for k1, b in product(k1_values, b_values):
        yield {"method": "bm25", "k1": k1, "b": b}
    for k1, b, weight, rrf_k in product(k1_values, b_values, bm25_weights, rrf_values):
        yield {"method": "hybrid", "k1": k1, "b": b,
               "dense_weight": 1.0, "bm25_weight": weight, "rrf_k": rrf_k}


def metrics_for_context(payload, judgments):
    result = evaluate(payload, judgments)
    cutoff = max(1, max(len(q["chunks"]) for q in payload["questions"]))
    full = evaluate(payload, judgments, (cutoff,))
    result["macro_full_context_document_coverage"] = full[
        "macro_at_chunk_k_document_coverage"][str(cutoff)]
    result["full_context_by_question"] = {
        q["id"]: q["at_chunk_k_document_coverage"][str(cutoff)] for q in full["questions"]}
    return result


def evaluate_configuration(config, questions, judgments, chunks, similarities,
                           lexical, tokenizer, token_counts, output):
    started = time.perf_counter()
    name = experiment_name(config)
    ranking = {"variant": name, "questions": []}
    context = {"variant": name, "questions": []}
    chunk_counts, token_totals = [], []
    for question in questions:
        text, qid = question["question"], str(question["id"])
        rows = rank_chunks(chunks, similarities[qid] if config["method"] != "bm25" else None,
                           text, config["method"], lexical,
                           config.get("rrf_k", 60), config.get("dense_weight", 1),
                           config.get("bm25_weight", 1))
        result = select_context(rows, tokenizer, token_counts=token_counts)
        base = {"id": question["id"], "question": text, "domain": question["domain"]}
        # Complete rankings retain every ID and score; text lives in shared catalog.
        ranking["questions"].append({**base, "chunk_ids": [c["chunk_id"] for c in rows],
                                     "scores": [c["score"] for c in rows]})
        selected = result["selected_chunks"]
        compact = [{k: v for k, v in c.items() if k not in ("text", "source_path")}
                   for c in selected]
        tokens = sum(len(tokenizer.encode(c["text"])) if c["truncated"]
                     else token_counts[c["chunk_id"]] for c in selected)
        context["questions"].append({**base, "chunks": compact,
                                     "context": result["context"], "chunk_text_tokens": tokens})
        chunk_counts.append(len(selected))
        token_totals.append(tokens)
    ranking_metrics = evaluate(ranking, judgments)
    context_metrics = metrics_for_context(context, judgments)
    for file, payload in (("ranking", ranking), ("context", context),
                          ("ranking-metrics", ranking_metrics), ("context-metrics", context_metrics)):
        write_json(output / f"{file}.json", payload)
    summary = {"name": name, **config, "question_count": len(questions),
               "ranking_ndcg_at10": ranking_metrics["macro_at_document_k"]["10"]["ndcg"]["value"],
               "context_ndcg_at10": context_metrics["macro_at_document_k"]["10"]["ndcg"]["value"],
               "context_recall_ge2": context_metrics["macro_full_context_document_coverage"]["recall_ge2"]["value"],
               "context_recall_ge1": context_metrics["macro_full_context_document_coverage"]["recall_ge1"]["value"],
               "mean_context_chunks": sum(chunk_counts) / len(questions),
               "mean_chunk_text_tokens": sum(token_totals) / len(questions),
               "max_chunk_text_tokens": max(token_totals),
               "incremental_replay_seconds": time.perf_counter() - started}
    for domain in sorted({q["domain"] for q in questions}):
        qids = [str(q["id"]) for q in questions if q["domain"] == domain]
        coverage = context_metrics["full_context_by_question"]
        summary[f"context_recall_ge2_{domain}"] = sum(coverage[qid]["recall_ge2"] for qid in qids) / len(qids)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k1", type=float, nargs="+", default=[.75, 1.25, 1.75])
    parser.add_argument("--b", type=float, nargs="+", default=[.5, .75, 1.0])
    parser.add_argument("--bm25-weight", type=float, nargs="+", default=[1.0, .75, .5])
    parser.add_argument("--rrf-k", type=float, nargs="+", default=[30, 60, 90])
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; use a new sweep name")
    configs = list(configurations(args.k1, args.b, args.bm25_weight, args.rrf_k))
    names = [experiment_name(c) for c in configs]
    if len(set(names)) != len(names):
        parser.error("Grid contains duplicate configurations or colliding names")
    started = time.perf_counter()
    questions, judgments, manifest = prepare(args.corpus)
    chunks, similarities, source_manifest = load_frozen_run(args.source_run, questions, judgments)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    token_counts = {c["chunk_id"]: len(tokenizer.encode(c["text"])) for c in chunks}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "chunks.jsonl"
    with catalog_path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps({**chunk, "token_count": token_counts[chunk["chunk_id"]]},
                                    ensure_ascii=False) + "\n")
    write_json(args.output_dir / "questions.json", questions)
    write_json(args.output_dir / "judgments.json", judgments)
    provenance = {"source_ranking": fingerprint(args.source_run / "ranking.json"),
                  "shared_catalog": fingerprint(catalog_path),
                  "judgments": fingerprint(args.output_dir / "judgments.json"),
                  "tokenizer_files": {p.name: fingerprint(p) for p in args.tokenizer.iterdir() if p.is_file()},
                  "code": {name: fingerprint(ROOT / name) for name in
                           ("src/retrieval.py", "tools/sweep_retrieval.py", "tools/evaluate_retrieval.py")}}
    manifest.update({"mode": "weighted_rrf_grid_frozen_replay", "sources": {**manifest["sources"], **provenance},
                     "source_manifest": source_manifest, "chunk_count": len(chunks),
                     "run_count": len(configs), "context_chunk_text_budget": MAX_CONTEXT_TOKENS,
                     "grid": configs, "lexical_votes": "positive BM25 scores only",
                     "lexical_tokenization": "Unicode word regex, casefold; no stemming/stopwords",
                     "timing_scope": "Frozen replay: excludes PDF parsing, dense embeddings and LLM inference"})
    write_json(args.output_dir / "manifest.json", manifest)
    summaries = []
    lexical_key, lexical = None, None
    for i, config in enumerate(configs, 1):
        key = (config.get("k1"), config.get("b"))
        if config["method"] != "dense" and lexical_key != key:
            tick = time.perf_counter()
            model = BM25Retriever(chunks, *key)
            cached = {q["question"]: model.scores(q["question"]) for q in questions}
            lexical = SimpleNamespace(scores=cached.__getitem__)
            lexical_key = key
            bm25_build_seconds = time.perf_counter() - tick
        name = experiment_name(config)
        summary = evaluate_configuration(config, questions, judgments, chunks, similarities,
                                         lexical, tokenizer, token_counts, args.output_dir / name)
        summaries.append(summary)
        write_json(args.output_dir / name / "manifest.json", {
            "status": "complete", "config": config, "question_count": len(questions),
            "sources": provenance, "context_chunk_text_budget": MAX_CONTEXT_TOKENS,
            "bm25_build_and_all_query_scores_seconds": bm25_build_seconds if lexical_key else None,
            "timing_scope": manifest["timing_scope"]})
        write_json(args.output_dir / "summary.json", summaries)
        print(f"[{i}/{len(configs)}] {name}: nDCG@10={summary['ranking_ndcg_at10']:.4f}; "
              f"context recall2={summary['context_recall_ge2']:.4f}", flush=True)
    columns = list(dict.fromkeys(key for row in summaries for key in row))
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    manifest.update({"status": "complete", "elapsed_seconds": time.perf_counter() - started,
                     "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "rss_scope": "Linux process only; excludes descendants and GPU"})
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
