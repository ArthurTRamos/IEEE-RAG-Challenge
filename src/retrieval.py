"""Dense baseline retrieval, shared by submission and local evaluation.

No generative model is imported or loaded here. Chunking and context selection
retain the starter kit's character-based overlap and first-overflow stopping rule.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_CONTEXT_TOKENS = 10_000
EMBED_MODEL_PATH = Path(__file__).parent / "embed_model"
DOCLING_MODELS_PATH = Path(__file__).parent / "docling_models"


@dataclass
class RetrievalIndex:
    chunks: list[dict]
    embeddings: np.ndarray
    embedder: object
    documents: list[str]
    failures: list[dict]
    build_seconds: float


def discover_pdfs(corpus_dir):
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"Corpus directory does not exist: {root}")
    paths = sorted(root.rglob("*.pdf"))
    if not paths:
        raise ValueError(f"No PDFs found under {root}")
    ids = [path.stem for path in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("PDF stems must be unique document IDs")
    return paths


def _chunk_markdown(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_index(corpus_dir, embed_model_path=EMBED_MODEL_PATH,
                docling_models_path=DOCLING_MODELS_PATH):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    pdfs = discover_pdfs(corpus_dir)
    options = PdfPipelineOptions(artifacts_path=str(docling_models_path),
                                 do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    embedder = SentenceTransformer(str(embed_model_path), device="cpu",
                                  local_files_only=True)
    chunks, failures = [], []
    for pdf in pdfs:
        try:
            text = converter.convert(str(pdf)).document.export_to_markdown()
            parts = _chunk_markdown(text)
            if not parts:
                raise ValueError("PDF produced no text chunks")
        except Exception as exc:
            failures.append({"document_id": pdf.stem, "error": str(exc)})
            print(f"[rag] skipping {pdf.name}: {exc}", flush=True)
            continue
        for number, part in enumerate(parts):
            chunks.append({"chunk_id": f"{pdf.stem}.pdf#{number}",
                           "document_id": pdf.stem,
                           "source_path": pdf.relative_to(corpus_dir).as_posix(),
                           "text": part})
    if not chunks:
        raise ValueError("Corpus produced no text chunks")
    embeddings = embedder.encode([chunk["text"] for chunk in chunks],
                                normalize_embeddings=True,
                                show_progress_bar=False, batch_size=64)
    return RetrievalIndex(chunks, np.asarray(embeddings, dtype=np.float32),
                          embedder, [pdf.stem for pdf in pdfs], failures,
                          time.perf_counter() - started)


def retrieve(question, index, tokenizer=None, max_tokens=MAX_CONTEXT_TOKENS):
    """Return the full ranking and, when tokenized, the actual baseline context.

    The budget counts chunk text only, as in the starter kit. Headers, question
    and chat template are not included. The oversized-first-chunk fallback is
    the original approximate character truncation, not a strict token limit.
    """
    if max_tokens <= 0:
        raise ValueError("Context budget must be positive")
    query = index.embedder.encode([question], normalize_embeddings=True,
                                  show_progress_bar=False)[0]
    similarities = index.embeddings @ query
    order = np.argsort(-similarities)
    ranking = [{**index.chunks[int(i)], "rank": rank,
                "score": float(similarities[i])}
               for rank, i in enumerate(order, 1)]
    if tokenizer is None:
        return {"chunks": ranking, "selected_chunks": None, "context": None}
    selected, used = [], 0
    for chunk in ranking:
        n = len(tokenizer.encode(chunk["text"]))
        if used + n > max_tokens:
            if not selected and n > 0:
                text = chunk["text"][:int(len(chunk["text"]) * max_tokens / n)]
                selected.append({**chunk, "text": text, "truncated": True})
            break
        selected.append({**chunk, "truncated": False})
        used += n
    body = "\n\n".join(f"[Fragment {i}]\n{c['text']}"
                         for i, c in enumerate(selected, 1))
    context = (f"Context extracted from a blind search over the corpus:\n\n{body}\n\n---\n\n"
               if selected else "")
    return {"chunks": ranking, "selected_chunks": selected, "context": context}
