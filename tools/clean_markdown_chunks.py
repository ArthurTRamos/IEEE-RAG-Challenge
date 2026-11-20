#!/usr/bin/env python3
"""Clean a prior Markdown export and regenerate chunks without rerunning PDFs.

Preserves raw Markdown and document metadata. Old chunk IDs/qrels must not be
reused as judgments for the newly generated chunks.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation import fingerprint, write_json
from src.retrieval import markdown_cleanup, _chunk_markdown, CHUNK_SIZE, CHUNK_OVERLAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/runs/pdf-markdown-chunks")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    started = time.perf_counter()
    original = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise ValueError("Source export is incomplete")
    docs = original["documents"]
    # Validate every input before producing output.
    for doc in docs:
        if fingerprint(args.source / doc["markdown_path"])["sha256"] != doc["markdown_sha256"]:
            raise ValueError(f"Source Markdown changed: {doc['document_id']}")
    args.output_dir.mkdir(parents=True)
    result_docs, removed_total = [], 0
    for doc in docs:
        raw = (args.source / doc["markdown_path"]).read_text(encoding="utf-8")
        cleaned = markdown_cleanup(raw)
        docid = doc["document_id"]
        raw_path = args.output_dir / "markdown" / f"{docid}.md"
        index_path = args.output_dir / "index-markdown" / f"{docid}.md"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # read_text normalizes CRLF/CR; copy the raw artifact as bytes so its
        # source hash and exact line endings remain unchanged.
        raw_path.write_bytes((args.source / doc["markdown_path"]).read_bytes())
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(cleaned["text"], encoding="utf-8")
        parts = _chunk_markdown(cleaned["text"])
        if not parts:
            raise ValueError(f"No chunks after cleanup: {docid}")
        chunks = []
        for number, part in enumerate(parts):
            path = args.output_dir / "chunks" / docid / f"chunk-{number:04}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(part, encoding="utf-8")
            chunks.append({"chunk_id": f"{docid}.pdf#{number}",
                           "path": path.relative_to(args.output_dir).as_posix(),
                           "sha256": fingerprint(path)["sha256"], "characters": len(part)})
        removed_total += len(raw) - len(cleaned["text"])
        result_docs.append({**doc,
            "markdown_path": raw_path.relative_to(args.output_dir).as_posix(),
            "markdown_sha256": fingerprint(raw_path)["sha256"],
            "index_markdown_path": index_path.relative_to(args.output_dir).as_posix(),
            "index_markdown_sha256": fingerprint(index_path)["sha256"],
            "cleanup": {k: v for k, v in cleaned.items() if k != "text"},
            "previous_chunk_count": doc["chunk_count"], "chunk_count": len(chunks), "chunks": chunks})
    manifest = {
        "status": "complete", "stage": "cleanup_and_rechunk_saved_markdown",
        "source_export": fingerprint(args.source / "manifest.json"),
        "metadata_source": original["metadata_source"],
        "code": [fingerprint(Path(__file__)), fingerprint(ROOT / "src/retrieval.py")],
        "config": {"chunk_size_characters": CHUNK_SIZE, "chunk_overlap_characters": CHUNK_OVERLAP,
                   "cleanup": "front_matter_before_first_body_marker_and_bibliography_section"},
        "document_count": len(result_docs), "previous_chunk_count": original["chunk_count"],
        "chunk_count": sum(d["chunk_count"] for d in result_docs),
        "removed_characters": removed_total,
        "documents": result_docs, "failures": [],
        "chunk_judgments_compatible_with_source": False,
        "gpu_used": False, "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"documents", "code", "config"}}, indent=2))


if __name__ == "__main__":
    main()
