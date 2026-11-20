#!/usr/bin/env python3
"""Export PDF Markdown and per-document chunks with the submission's functions."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_evaluation import ROOT, DEFAULT_JUDGMENTS, fingerprint, write_json

sys.path.insert(0, str(ROOT))
from src.retrieval import (CHUNK_SIZE, CHUNK_OVERLAP, DOCLING_MODELS_PATH,
                           discover_pdfs, _export_markdown, _chunk_markdown, markdown_cleanup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/runs/pdf-markdown-chunks")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--document-id", action="append",
                        help="Export only this document ID; may be repeated")
    parser.add_argument("--pdf-backend", choices=("default", "serial-docling-parse", "pypdfium2"),
                        default="default", help="Alternative parser for problematic PDFs")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; choose a new directory")

    # Only local artifacts are allowed, even when running on a connected host.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    started = time.perf_counter()
    pdfs = discover_pdfs(args.corpus)
    if args.document_id:
        requested = set(args.document_id)
        missing = requested - {p.stem for p in pdfs}
        if missing:
            parser.error(f"Document IDs not found: {sorted(missing)}")
        pdfs = [p for p in pdfs if p.stem in requested]
    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    catalog = {d["document_id"]: d for d in source["corpus_documents"]}
    sources = {}
    for pdf in pdfs:
        if pdf.stem not in catalog:
            raise ValueError(f"Missing document metadata: {pdf}")
        sources[pdf.stem] = fingerprint(pdf)
        if sources[pdf.stem]["sha256"] != catalog[pdf.stem]["pdf_sha256"]:
            raise ValueError(f"PDF hash differs from judged document: {pdf}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    options = PdfPipelineOptions(
        artifacts_path=str(DOCLING_MODELS_PATH), do_ocr=False,
        do_table_structure=False, accelerator_options=AcceleratorOptions(device=device))
    format_options = PdfFormatOption(pipeline_options=options)
    if args.pdf_backend == "serial-docling-parse":
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        format_options = PdfFormatOption(pipeline_options=options,
                                         backend=DoclingParseDocumentBackend)
    elif args.pdf_backend == "pypdfium2":
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        format_options = PdfFormatOption(pipeline_options=options,
                                         backend=PyPdfiumDocumentBackend)
    converter = DocumentConverter(format_options={InputFormat.PDF: format_options})
    manifest = {
        "status": "running", "metadata_source": fingerprint(args.metadata),
        "code": [fingerprint(ROOT / p) for p in
                 ("src/retrieval.py", "tools/export_pdf_chunks.py")],
        "config": {"device": device,
                   "pdf_backend": args.pdf_backend,
                   "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "do_ocr": False, "do_table_structure": False,
                   "pdfplumber_repairs": True,
                   "markdown_cleanup": "front_matter_and_bibliography_sections",
                   "chunk_size_characters": CHUNK_SIZE,
                   "chunk_overlap_characters": CHUNK_OVERLAP,
                   "chunk_size_is_hard_limit": False},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("docling", "docling-core", "pdfplumber", "torch")},
        "documents": [], "failures": [],
        "missing_metadata_documents": sorted(set(catalog) - {p.stem for p in pdfs}),
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Exporting {len(pdfs)} PDFs on {device}", flush=True)
    for position, pdf in enumerate(pdfs, 1):
        tick = time.perf_counter()
        try:
            result = converter.convert(str(pdf))
            status = result.status.value
            if status != "success":
                raise RuntimeError(f"Docling conversion status: {status}; {result.errors}")
            text = _export_markdown(result.document, pdf)
            md = args.output_dir / "markdown" / f"{pdf.stem}.md"
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(text, encoding="utf-8")
            cleaned = markdown_cleanup(md.read_text(encoding="utf-8"))
            index_md = args.output_dir / "index-markdown" / f"{pdf.stem}.md"
            index_md.parent.mkdir(parents=True, exist_ok=True)
            index_md.write_text(cleaned["text"], encoding="utf-8")
            parts = _chunk_markdown(cleaned["text"])
            if not parts:
                raise ValueError("PDF produced no text chunks")
            chunk_dir = args.output_dir / "chunks" / pdf.stem
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunks = []
            for number, part in enumerate(parts):
                target = chunk_dir / f"chunk-{number:04d}.md"
                target.write_text(part, encoding="utf-8")
                chunks.append({"chunk_id": f"{pdf.stem}.pdf#{number}",
                               "path": target.relative_to(args.output_dir).as_posix(),
                               "sha256": fingerprint(target)["sha256"],
                               "characters": len(part)})
            manifest["documents"].append({
                "document_id": pdf.stem, "source_metadata": catalog[pdf.stem],
                "source_pdf": sources[pdf.stem],
                "source_path": pdf.relative_to(args.corpus).as_posix(),
                "markdown_path": md.relative_to(args.output_dir).as_posix(),
                "markdown_sha256": fingerprint(md)["sha256"],
                "index_markdown_path": index_md.relative_to(args.output_dir).as_posix(),
                "index_markdown_sha256": fingerprint(index_md)["sha256"],
                "cleanup": {key: value for key, value in cleaned.items() if key != "text"},
                "pages": len(result.document.pages), "conversion_status": status,
                "chunk_count": len(parts), "chunks": chunks,
                "elapsed_seconds": time.perf_counter() - tick})
            print(f"[{position}/{len(pdfs)}] {pdf.stem}: {len(parts)} chunks, "
                  f"{time.perf_counter() - tick:.1f}s", flush=True)
        except Exception as exc:
            manifest["failures"].append({"document_id": pdf.stem, "error": str(exc)})
            print(f"[{position}/{len(pdfs)}] FAILED {pdf.stem}: {exc}", flush=True)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(manifest_path, manifest)
    manifest["status"] = "failed" if manifest["failures"] else "complete"
    manifest["document_count"] = len(manifest["documents"])
    manifest["chunk_count"] = sum(d["chunk_count"] for d in manifest["documents"])
    manifest["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if device == "cuda":
        manifest["torch_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["torch_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(manifest_path, manifest)
    print(f"{manifest['status']}: {manifest['document_count']} documents, "
          f"{manifest['chunk_count']} chunks -> {args.output_dir}", flush=True)
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
