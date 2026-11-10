#!/usr/bin/env python3
"""Run and evaluate the dense baseline without loading generative LLM weights."""

import argparse
import resource
import sys
import time
from pathlib import Path

from evaluate_retrieval import evaluate
from prepare_evaluation import (ROOT, DEFAULT_QUESTIONS, DEFAULT_JUDGMENTS,
                                prepare, write_json, fingerprint)

sys.path.insert(0, str(ROOT))
from src.retrieval import (build_index, retrieve, EMBED_MODEL_PATH,
                           DOCLING_MODELS_PATH, MAX_CONTEXT_TOKENS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--embed-model", type=Path, default=EMBED_MODEL_PATH)
    parser.add_argument("--docling-models", type=Path, default=DOCLING_MODELS_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer", type=Path, help="Local OLMo tokenizer directory; no LLM weights needed")
    mode.add_argument("--ranking-only", action="store_true", help="Skip context selection; no context metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; use a new directory for each run")
    started = time.perf_counter()
    try:
        questions, judgments, manifest = prepare(args.corpus, args.questions, args.judgments)
        for path in (args.embed_model, args.docling_models, args.tokenizer):
            if path is not None and not path.is_dir():
                raise ValueError(f"Local model directory does not exist: {path}")
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
        index = build_index(args.corpus, args.embed_model, args.docling_models)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    write_json(args.output_dir / "questions.json", questions)
    write_json(args.output_dir / "judgments.json", judgments)
    manifest.update({"parse_failures": index.failures, "index_build_seconds": index.build_seconds,
                     "chunk_count": len(index.chunks), "embedding_bytes": index.embeddings.nbytes,
                     "config": {"embedding_model": str(args.embed_model.resolve()),
                                "docling_models": str(args.docling_models.resolve()),
                                "tokenizer": str(args.tokenizer.resolve()) if args.tokenizer else None,
                                "chunk_size_characters": 800, "chunk_overlap_characters": 150,
                                "context_chunk_text_budget": MAX_CONTEXT_TOKENS}})
    ranking = {"variant": "starter_dense", "questions": []}
    context = {"variant": "starter_dense_selected_context", "questions": []}
    for question in questions:
        tick = time.perf_counter()
        result = retrieve(question["question"], index, tokenizer)
        base = {"id": question["id"], "domain": question["domain"],
                "question": question["question"]}
        ranking["questions"].append({**base, "chunks": result["chunks"],
                                     "retrieval_seconds": time.perf_counter() - tick})
        if tokenizer is not None:
            context["questions"].append({**base, "chunks": result["selected_chunks"],
                                        "context": result["context"]})
        print(f"[eval] question {question['id']} complete", flush=True)
    outputs = [("ranking", ranking)]
    if tokenizer is not None:
        outputs.append(("context", context))
    for name, payload in outputs:
        path = args.output_dir / f"{name}.json"
        write_json(path, payload)
        metrics = evaluate(payload, judgments)
        metrics["sources"] = {"input": fingerprint(path),
                              "judgments": fingerprint(args.output_dir / "judgments.json")}
        write_json(args.output_dir / f"{name}-metrics.json", metrics)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    # Linux reports KiB; includes this process, not descendants or GPU memory.
    manifest["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    manifest["process_peak_rss_scope"] = "Linux process RSS only; excludes descendants and GPU memory"
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Evaluated {len(questions)} questions -> {args.output_dir}")


if __name__ == "__main__":
    main()
