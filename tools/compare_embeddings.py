#!/usr/bin/env python3
"""Compare MiniLM and Arctic Embed XS on one extraction of the public corpus."""

import argparse
from importlib.metadata import version
from pathlib import Path
import resource
import sys
import time

from evaluate_retrieval import evaluate
from prepare_evaluation import ROOT, fingerprint, prepare, write_json

sys.path.insert(0, str(ROOT))
from src.main import (BM25_B, BM25_K1, RETRIEVAL_METHOD, RRF_BM25_WEIGHT,
                      RRF_DENSE_WEIGHT, RRF_K)
from src.retrieval import (BM25Retriever, EMBED_MODEL_PATH, encode_chunks,
                           extract_corpus, rank_chunks, select_context)


def evaluate_encoder(encoder, prompt, limit, chunks, questions, judgments,
                     tokenizer, token_counts, lexical, output):
    encoder.max_seq_length = limit
    tick = time.perf_counter()
    embeddings = encode_chunks(chunks, encoder)
    embedding_seconds = time.perf_counter() - tick
    rankings, contexts = [], []
    tick = time.perf_counter()
    for question in questions:
        query = question["question"]
        vector = encoder.encode([query], prompt_name=prompt, normalize_embeddings=True,
                                show_progress_bar=False)[0]
        ranking = rank_chunks(chunks, embeddings @ vector, query, RETRIEVAL_METHOD,
                              lexical, RRF_K, RRF_DENSE_WEIGHT, RRF_BM25_WEIGHT)
        context = select_context(ranking, tokenizer, token_counts=token_counts)
        base = {key: question[key] for key in ("id", "domain", "question")}
        rankings.append({**base, "chunk_ids": [row["chunk_id"] for row in ranking]})
        contexts.append({**base, "chunks": context["selected_chunks"], "context": context["context"]})
    retrieval_seconds = time.perf_counter() - tick

    ranking_payload = {"questions": rankings}
    context_payload = {"questions": contexts}
    ranking_metrics = evaluate(ranking_payload, judgments)
    full_cutoff = max(10, max(len(row["chunks"]) for row in contexts))
    context_metrics = evaluate(context_payload, judgments, (10, full_cutoff)
                               if full_cutoff > 10 else (10,))
    for name, payload in (("ranking", ranking_payload), ("context", context_payload),
                          ("ranking-metrics", ranking_metrics), ("context-metrics", context_metrics)):
        write_json(output / f"{name}.json", payload)
    context_coverage = context_metrics["macro_at_chunk_k_document_coverage"][str(full_cutoff)]
    return {
        "ndcg_at_10_documents": ranking_metrics["macro_at_document_k"]["10"]["ndcg"]["value"],
        "recall_grade_2_at_10_chunks": context_metrics["macro_at_chunk_k_document_coverage"]["10"]["recall_ge2"]["value"],
        "recall_grade_2_full_context": context_coverage["recall_ge2"]["value"],
        "recall_grades_1_and_2_full_context": context_coverage["recall_ge1"]["value"],
        "embedding_seconds": embedding_seconds, "retrieval_seconds": retrieval_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--arctic-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    for path in (args.arctic_model, args.tokenizer):
        if not path.is_dir():
            parser.error(f"Model directory does not exist: {path}")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    started = time.perf_counter()
    questions, judgments, manifest = prepare(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    print(f"[compare] Extracting {manifest['document_count']} PDFs", flush=True)
    tick = time.perf_counter()
    chunks, _, failures = extract_corpus(args.corpus)
    extraction_seconds = time.perf_counter() - tick
    print(f"[compare] Extracted {len(chunks)} chunks in {extraction_seconds:.1f}s", flush=True)
    write_json(args.output_dir / "chunks.json", chunks)
    write_json(args.output_dir / "judgments.json", judgments)
    manifest.update({
        "chunk_count": len(chunks), "parse_failures": failures,
        "extraction_seconds": extraction_seconds,
        "retrieval": {"method": RETRIEVAL_METHOD, "k1": BM25_K1, "b": BM25_B,
                      "rrf_k": RRF_K, "dense_weight": RRF_DENSE_WEIGHT,
                      "bm25_weight": RRF_BM25_WEIGHT},
        "versions": {package: version(package) for package in
                     ("torch", "transformers", "sentence-transformers", "docling", "spacy")},
    })
    manifest["sources"].update({"chunks": fingerprint(args.output_dir / "chunks.json"),
                                "retrieval": fingerprint(ROOT / "src/retrieval.py"),
                                "experiment": fingerprint(__file__)})
    write_json(args.output_dir / "manifest.json", manifest)
    token_counts = {c["chunk_id"]: len(tokenizer.encode(c["text"])) for c in chunks}
    tick = time.perf_counter()
    bm25 = BM25Retriever(chunks, BM25_K1, BM25_B)
    cached_scores = {q["question"]: bm25.scores(q["question"]) for q in questions}
    manifest["bm25_seconds"] = time.perf_counter() - tick

    class LexicalScores:
        scores = staticmethod(cached_scores.__getitem__)

    results = []
    for name, path, prompt in (("minilm-l6-v2", EMBED_MODEL_PATH, None),
                               ("arctic-embed-xs", args.arctic_model, "query")):
        encoder = SentenceTransformer(str(path), device=None, local_files_only=True)
        weight = fingerprint(path / "model.safetensors")
        lengths = [len(row) for row in encoder.tokenizer(
            [c["text"] for c in chunks], truncation=False, verbose=False)["input_ids"]]
        for limit in (256, 512):
            variant = f"{name}-{limit}"
            print(f"[compare] Running {variant} on {encoder.device}", flush=True)
            metrics = evaluate_encoder(encoder, prompt, limit, chunks, questions, judgments,
                                       tokenizer, token_counts, LexicalScores(),
                                       args.output_dir / variant)
            results.append({"variant": variant, "max_input_tokens": limit,
                            "query_prompt": encoder.prompts[prompt] if prompt else None,
                            "encoder_weights": weight, "device": str(encoder.device),
                            "chunks_over_limit": sum(length > limit for length in lengths),
                            **metrics})
            write_json(args.output_dir / "summary.json", results)
            print(f"[compare] {variant} nDCG@10={metrics['ndcg_at_10_documents']:.4f}", flush=True)
        del encoder

    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
