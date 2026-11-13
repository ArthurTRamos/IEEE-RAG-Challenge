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
    lexical_index: object = None


def lexical_tokens(text):
    """Unicode words/numbers, case-folded; no stemming or stopword resources."""
    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    def __init__(self, chunks, k1=1.25, b=0.75):
        from rank_bm25 import BM25Okapi

        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        tokens = [lexical_tokens(chunk["text"]) for chunk in chunks]
        if not any(tokens):
            raise ValueError("BM25 corpus has no lexical tokens")
        self.model = BM25Okapi(tokens, k1=k1, b=b)

    def scores(self, question):
        # rank-bm25 divides 0/0 for lexically empty chunks when b=1. Their
        # contribution is zero: no query term occurs. Keep such chunks in the
        # corpus statistics so existing configurations retain identical scores.
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.asarray(self.model.get_scores(lexical_tokens(question)))
        values[np.asarray(self.model.doc_len) == 0] = 0.0
        if not np.isfinite(values).all():
            raise ValueError("BM25 produced non-finite scores for nonempty chunks")
        return values


def rank_chunks(chunks, dense_scores, question, method="dense", lexical_index=None,
                rrf_k=60, dense_weight=1.0, bm25_weight=1.0):
    """Weighted RRF over all dense chunks and positive BM25 matches.

    No fixed top-k candidate cutoff. BM25 zero/negative scores contribute no
    lexical vote. Stable lexical/fusion ties follow canonical chunk order.
    The dense sort retains the original NumPy default for baseline equivalence.
    """
    if method not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unknown retrieval method: {method}")
    if not np.isfinite(rrf_k) or rrf_k <= 0:
        raise ValueError("RRF constant must be finite and positive")
    if (not np.isfinite([dense_weight, bm25_weight]).all()
            or min(dense_weight, bm25_weight) < 0 or dense_weight + bm25_weight <= 0):
        raise ValueError("RRF weights must be finite, nonnegative and not both zero")
    lexical_scores = None
    if method != "dense":
        if lexical_index is None:
            raise ValueError("BM25 index is required for lexical/hybrid retrieval")
        lexical_scores = lexical_index.scores(question)
    if method == "dense":
        scores = np.asarray(dense_scores)
        order = np.argsort(-scores)
    elif method == "bm25":
        scores = lexical_scores
        order = np.argsort(-scores, kind="stable")
    else:
        scores = np.zeros(len(chunks), dtype=np.float64)
        dense_order = np.argsort(-np.asarray(dense_scores))
        lexical_order = np.argsort(-lexical_scores, kind="stable")
        positive_order = lexical_order[lexical_scores[lexical_order] > 0]
        for channel, weight in ((dense_order, dense_weight), (positive_order, bm25_weight)):
            scores[channel] += weight / (rrf_k + np.arange(1, len(channel) + 1))
        order = np.argsort(-scores, kind="stable")
    ranking = []
    for rank, i in enumerate(order, 1):
        row = {**chunks[int(i)], "rank": rank, "score": float(scores[i])}
        if dense_scores is not None:
            row["dense_score"] = float(dense_scores[i])
        if lexical_scores is not None:
            row["bm25_score"] = float(lexical_scores[i])
        ranking.append(row)
    return ranking


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


def retrieve(question, index, tokenizer=None, max_tokens=MAX_CONTEXT_TOKENS,
             method="dense", k1=1.25, b=0.75, rrf_k=60,
             dense_weight=1.0, bm25_weight=1.0, lexical_question=None):
    """Return the full ranking and, when tokenized, the actual baseline context.

    The budget counts chunk text only, as in the starter kit. Headers, question
    and chat template are not included. The oversized-first-chunk fallback is
    the original approximate character truncation, not a strict token limit.
    """
    similarities = None
    if method != "bm25":
        query = index.embedder.encode([question], normalize_embeddings=True,
                                  show_progress_bar=False)[0]
        similarities = index.embeddings @ query
    if method != "dense" and (index.lexical_index is None or
                               (index.lexical_index.k1, index.lexical_index.b) != (k1, b)):
        index.lexical_index = BM25Retriever(index.chunks, k1, b)
    lexical = index.lexical_index
    if lexical_question is not None and method != "dense":
        from types import SimpleNamespace
        lexical = SimpleNamespace(scores=lambda _: index.lexical_index.scores(lexical_question))
    ranking = rank_chunks(index.chunks, similarities, question, method,
                          lexical, rrf_k, dense_weight, bm25_weight)
    return select_context(ranking, tokenizer, max_tokens)


def select_context(ranking, tokenizer=None, max_tokens=MAX_CONTEXT_TOKENS,
                   token_counts=None):
    """Apply the same baseline context policy to any ranked chunk list."""
    if max_tokens <= 0:
        raise ValueError("Context budget must be positive")
    if tokenizer is None:
        return {"chunks": ranking, "selected_chunks": None, "context": None}
    selected, used = [], 0
    for chunk in ranking:
        n = token_counts.get(chunk["chunk_id"]) if token_counts is not None else None
        if n is None:
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
