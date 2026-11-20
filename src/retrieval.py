"""Dense baseline retrieval, shared by submission and local evaluation.

No generative model is imported or loaded here. Chunking and context selection
retain the starter kit's character-based overlap and first-overflow stopping rule.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_CONTEXT_TOKENS = 10_000
EMBED_MODEL_PATH = Path(__file__).parent / "embed_model"
DOCLING_MODELS_PATH = Path(__file__).parent / "docling_models"
SPACY_MODEL_PATH = Path(__file__).parent / "spacy_model"  # en_core_web_sm 3.8.0


@dataclass
class RetrievalIndex:
    chunks: list[dict]
    embeddings: np.ndarray
    embedder: object
    documents: list[str]
    failures: list[dict]
    build_seconds: float
    lexical_index: object = None


def _fold(text):
    """Case-fold and strip diacritics: 'Café' -> 'cafe'."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _nlp():
    """Tagger + lemmatizer only; parser and NER are not needed for lemmas."""
    import spacy
    return spacy.load(SPACY_MODEL_PATH, exclude=["parser", "ner"])


_NUMBER = r"[+\-−]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+\-−]?\d+)?"
_POWER = r"(?:\s*\^\s*(?:\{[+\-−]?\d+\}|[+\-−]?\d+)|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)"
_NAME = r"[^\W\d_]\w*"
# Match the longest expression before spaCy's punctuation/stopword handling.
# Boundaries prevent numeric suffixes inside ordinary identifiers being split.
TECHNICAL_TERM = re.compile(
    rf"(?<!\w)(?:{_NUMBER}(?:{_POWER}|/{_NUMBER})?(?:[a-zA-Z]+\b)?|"
    rf"{_NAME}(?:[-‐‑/]{_NAME})+(?:[+#]+)?|{_NAME}[+#]+|{_NAME}{_POWER})(?!\w)")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_TECH_DASHES = str.maketrans({"−": "-", "‐": "-", "‑": "-"})
_NEGATIONS = {"no", "not", "never", "without", "neither", "nor", "cannot", "n't"}


def _technical_term(text):
    # Preserve exponent semantics before NFKD turns superscripts into digits.
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
                  lambda m: "^" + m[0].translate(_SUPERSCRIPTS), text)
    text = re.sub(r"\s+|[{}]", "", text).translate(_TECH_DASHES)
    return _fold(text)


def _lexical_doc(doc, mode):
    if mode == "legacy":
        return [word for tok in doc if not tok.is_stop
                for word in re.findall(r"\w+", _fold(tok.lemma_))]
    spans = list(TECHNICAL_TERM.finditer(doc.text))
    index = 0
    result = []
    for tok in doc:
        start, end = tok.idx, tok.idx + len(tok.text)
        while index < len(spans) and spans[index].end() <= start:
            index += 1
        if index < len(spans) and spans[index].start() < end:
            # spaCy may split one expression across tokens, or put multiple
            # expressions in one token. Preserve all surrounding text too.
            cursor = start
            while index < len(spans) and spans[index].start() < end:
                span = spans[index]
                if cursor < span.start():
                    result.extend(re.findall(r"\w+", _fold(doc.text[cursor:span.start()])))
                if span.start() >= start:
                    result.append(_technical_term(span[0]))
                cursor = max(cursor, span.end())
                if span.end() > end:
                    break
                index += 1
            if cursor < end:
                result.extend(re.findall(r"\w+", _fold(doc.text[cursor:end])))
            continue
        negation = tok.lower_ in _NEGATIONS or tok.lemma_.lower() in _NEGATIONS
        if negation:
            result.append("not" if tok.lower_ == "n't" else _fold(tok.lower_))
        elif not tok.is_stop:
            result.extend(re.findall(r"\w+", _fold(tok.lemma_)))
    return result


def lexical_tokens_many(texts, mode="technical"):
    """Shared query/document representation; legacy is retained for ablations.

    Technical mode preserves compounds, signed numbers, ratios, exponents and
    method suffixes as complete tokens. Ordinary words retain spaCy lemmas and
    accent/case folding; negations survive stopword removal. This is lexical
    matching, not a mathematical parser or proof of logical equivalence.
    """
    if mode not in {"technical", "legacy"}:
        raise ValueError(f"Unknown lexical tokenization: {mode}")
    return [_lexical_doc(doc, mode) for doc in _nlp().pipe(texts, batch_size=256)]


def lexical_tokens(text, mode="technical"):
    return lexical_tokens_many([text], mode=mode)[0]


class BM25Retriever:
    def __init__(self, chunks, k1=1.25, b=0.75, tokenization="technical"):
        from rank_bm25 import BM25Okapi

        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.tokenization = tokenization
        tokens = lexical_tokens_many((chunk["text"] for chunk in chunks), mode=tokenization)
        if not any(tokens):
            raise ValueError("BM25 corpus has no lexical tokens")
        self.model = BM25Okapi(tokens, k1=k1, b=b)

    def scores(self, question):
        # rank-bm25 divides 0/0 for lexically empty chunks when b=1. Their
        # contribution is zero: no query term occurs. Keep such chunks in the
        # corpus statistics so existing configurations retain identical scores.
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.asarray(self.model.get_scores(lexical_tokens(question, mode=self.tokenization)))
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


# Docling without table structure collapses each table into one cell: "| ... |\n|---|".
ONE_CELL_TABLE = re.compile(r"^\|[^|\n]*\|\n\|-+\|$", re.M)
PLUMBER_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}
# Docling without formula enrichment exports each display formula as this placeholder.
FORMULA_GAP = "<!-- formula-not-decoded -->"
UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)|[\ue000-\uf8ff]")  # big operators with no Unicode


def _export_markdown(document, pdf_path):
    """Docling markdown; collapsed tables and undecoded formulas are re-read by
    pdfplumber inside Docling's bbox."""
    text = document.export_to_markdown()
    tables = [t for t in document.tables if t.data.num_cols <= 1]
    formulas = [t for t in document.texts if t.label == "formula" and t.orig and not t.text]
    if text.count(FORMULA_GAP) != len(formulas):
        formulas = []  # placeholders cannot be paired with items: leave them untouched
    if not (tables and ONE_CELL_TABLE.search(text)) and not formulas:
        return text
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pending = iter(tables)  # markdown and document items share reading order

        def replace(match):
            rows = _plumber_rows(pdf, next(pending, None))
            return _to_markdown(rows) if rows else match.group(0)
        text = ONE_CELL_TABLE.sub(replace, text)
        pending = iter(formulas)
        return re.sub(re.escape(FORMULA_GAP),
                      lambda m: _plumber_formula(pdf, next(pending)) or m.group(0), text)


def _crop(pdf, item):
    prov = item.prov[0]
    page = pdf.pages[prov.page_no - 1]
    box = prov.bbox.to_top_left_origin(page.height)
    return page.crop((box.l, box.t, box.r, box.b), strict=False)


def _plumber_rows(pdf, table):
    if table is None or not table.prov:
        return None
    table = _crop(pdf, table).find_table(PLUMBER_SETTINGS)
    # Same grid as extract_table; x_tolerance=1 only restores spaces inside small-font cells.
    rows = table.extract(x_tolerance=1) if table else None
    return rows if rows and any(cell for row in rows for cell in row) else None


def _plumber_formula(pdf, formula):
    if not formula.prov:
        return None
    raw = _crop(pdf, formula).extract_text(x_tolerance=1) or ""
    return " ".join(UNMAPPED_GLYPH.sub(" ", raw).split()) or None


def _to_markdown(rows):
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    rows.insert(1, ["---"] * len(rows[0]))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


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
    embedder = SentenceTransformer(str(embed_model_path), device=None,
                                  local_files_only=True)
    chunks, failures = [], []
    for pdf in pdfs:
        try:
            text = _export_markdown(converter.convert(str(pdf)).document, pdf)
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
             dense_weight=1.0, bm25_weight=1.0):
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
    ranking = rank_chunks(index.chunks, similarities, question, method,
                          index.lexical_index, rrf_k, dense_weight, bm25_weight)
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
