"""
RAG baseline — IEEE-CIS Technical Challenge on Domain-Specific Question
Answering with LLMs.

Blind retrieval over the whole corpus: every PDF in corpus_dir is parsed with
Docling, chunked and embedded once per run, then for each question every
chunk is ranked by cosine similarity and the best-ranked ones are packed
into a fixed context budget before asking the base model. No document_id is
known to this code — the same "search has to find it" design as a real
participant would face.

Needs a local copy of Docling's layout model and of the embedding model next
to this file (see build.sh): the evaluation environment has no internet
access, so neither can be downloaded there the first time they're used.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)

# Works both as an imported package and as the official top-level main.py.
if __package__:
    from .retrieval import build_index, retrieve
else:
    from retrieval import build_index, retrieve

MAX_NEW_TOKENS = 2048
# Selected submission: dense + BM25 with weighted RRF.
RETRIEVAL_METHOD = "hybrid"
BM25_K1 = 1.25
BM25_B = 0.75
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 0.75
RERANK_ENABLED = False  # Enabled only in the dedicated reranker submission build.
RERANK_CANDIDATES = 100
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}
_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is None:
        if __package__:
            from .reranking import ChunkReranker
        else:
            from reranking import ChunkReranker
        _reranker = ChunkReranker(
            Path(__file__).parent / "reranker_model", device="cpu",
            max_length=RERANK_MAX_LENGTH, batch_size=RERANK_BATCH_SIZE,
        )
    return _reranker


def _load(model_id: str, corpus_dir: str):
    key = (model_id, str(Path(corpus_dir).resolve()))
    if key not in _cache:
        index = build_index(corpus_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _cache[key] = (tokenizer, model, index)
    return _cache[key]


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    tokenizer, model, index = _load(model_id, corpus_dir)
    context = retrieve(question, index, tokenizer, method=RETRIEVAL_METHOD,
                       k1=BM25_K1, b=BM25_B, rrf_k=RRF_K,
                       dense_weight=RRF_DENSE_WEIGHT,
                       bm25_weight=RRF_BM25_WEIGHT,
                       reranker=_load_reranker() if RERANK_ENABLED else None,
                       rerank_candidates=RERANK_CANDIDATES)["context"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
