#!/usr/bin/env python3
"""Export the public corpus and its retrieval chunks as Markdown."""

import argparse
import json
from pathlib import Path
import sys
import time

from prepare_evaluation import DEFAULT_JUDGMENTS, ROOT, fingerprint, prepare, write_json

sys.path.insert(0, str(ROOT))
from src.retrieval import (CHUNK_OVERLAP, CHUNK_SIZE, DOCLING_MODELS_PATH,
                           _chunk_markdown, _export_markdown, _pdf_converter, discover_pdfs)


def convert_markdown(converter, pdf):
    from docling.datamodel.base_models import ConversionStatus

    result = converter.convert(pdf, raises_on_error=False)
    source_pages = result.input.page_count
    if (result.status != ConversionStatus.SUCCESS or not source_pages
            or set(result.document.pages) != set(range(1, source_pages + 1))):
        raise ValueError(f"{result.status.value}, {len(result.document.pages)}/{source_pages} "
                         f"pages: {result.errors}")
    text = _export_markdown(result.document, pdf)
    if not text.strip():
        raise ValueError("PDF produced empty Markdown")
    return text, source_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")

    import torch
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, evaluation = prepare(args.corpus)
    if evaluation["missing_document_ids"]:
        parser.error(f"Missing PDFs: {evaluation['missing_document_ids']}")
    metadata = json.loads(DEFAULT_JUDGMENTS.read_text(encoding="utf-8"))
    catalog = {row["document_id"]: row for row in metadata["corpus_documents"]}
    pdfs = discover_pdfs(args.corpus)
    (output / "markdown").mkdir(parents=True)
    converter = _pdf_converter(DOCLING_MODELS_PATH, device=device)
    fallback = None
    started = time.perf_counter()
    manifest = {
        "status": "extracting",
        "sources": {"judgments": fingerprint(DEFAULT_JUDGMENTS),
                    "retrieval": fingerprint(ROOT / "src/retrieval.py")},
        "config": {"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "do_ocr": False, "do_table_structure": False,
                   "chunk_size_characters": CHUNK_SIZE, "chunk_overlap_characters": CHUNK_OVERLAP},
        "documents": [], "failures": [],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Docling on {manifest['config']['gpu'] or device}", flush=True)

    for number, pdf in enumerate(pdfs, 1):
        source = catalog[pdf.stem]
        backend = "docling-parse"
        retry_reason = None
        try:
            text, pages = convert_markdown(converter, pdf)
        except Exception as exc:
            retry_reason = str(exc)
            backend = "pdfium"
            print(f"[{number}/{len(pdfs)}] {pdf.stem}: retrying with PDFium", flush=True)
            if fallback is None:
                fallback = _pdf_converter(DOCLING_MODELS_PATH, device=device,
                                          backend=PyPdfiumDocumentBackend)
            try:
                text, pages = convert_markdown(fallback, pdf)
            except Exception as retry_exc:
                manifest["failures"].append({"document_id": pdf.stem, "docling_parse": retry_reason,
                                             "pdfium": str(retry_exc)})
                write_json(manifest_path, manifest)
                continue

        markdown = Path("markdown") / f"{pdf.stem}.md"
        (output / markdown).write_text(text, encoding="utf-8")
        manifest["documents"].append({
            "document_id": pdf.stem, "title": source["title"], "domain": source["domain"],
            "pdf": pdf.relative_to(args.corpus).as_posix(), "pdf_sha256": source["pdf_sha256"],
            "pages": pages, "metadata_pages": source["pages"], "backend": backend,
            "retry_reason": retry_reason, "markdown": markdown.as_posix(),
            "markdown_sha256": fingerprint(output / markdown)["sha256"],
        })
        write_json(manifest_path, manifest)
        print(f"[{number}/{len(pdfs)}] {pdf.stem}: {pages} pages via {backend}", flush=True)

    manifest["extraction_seconds"] = time.perf_counter() - started
    manifest["status"] = "chunking"
    write_json(manifest_path, manifest)
    for document in manifest["documents"]:
        doc_id = document["document_id"]
        text = (output / document["markdown"]).read_text(encoding="utf-8")
        folder = Path("chunks") / doc_id
        (output / folder).mkdir(parents=True)
        document["chunks"] = []
        for number, chunk in enumerate(_chunk_markdown(text)):
            path = folder / f"chunk-{number:04d}.md"
            (output / path).write_text(chunk, encoding="utf-8")
            document["chunks"].append({"chunk_id": f"{doc_id}.pdf#{number}", "path": path.as_posix()})

    manifest["document_count"] = len(manifest["documents"])
    manifest["chunk_count"] = sum(len(doc["chunks"]) for doc in manifest["documents"])
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["status"] = "partial" if manifest["failures"] else "complete"
    write_json(manifest_path, manifest)
    print(f"{manifest['document_count']} documents and {manifest['chunk_count']} chunks -> {output}")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
