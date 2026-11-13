"""
DBpedia aliases experiment — IEEE-CIS Technical Challenge on Domain-Specific Question
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

import hashlib
import json
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
    from .kg_expansion import KnowledgeExpansion
else:
    from retrieval import build_index, retrieve
    from kg_expansion import KnowledgeExpansion

MAX_NEW_TOKENS = 2048
# Selected submission: dense + BM25 with weighted RRF.
RETRIEVAL_METHOD = "hybrid"
BM25_K1 = 1.25
BM25_B = 0.75
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 0.75

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}


def _load(model_id: str, corpus_dir: str):
    key = (model_id, str(Path(corpus_dir).resolve()))
    if key not in _cache:
        index = build_index(corpus_dir)
        config = json.loads((Path(__file__).parent / "kg_config.json").read_text(encoding="utf-8"))
        resource = Path(__file__).parent / "kg_resources/kg_rag_chunks.jsonl"
        if hashlib.sha256(resource.read_bytes()).hexdigest() != config["kg_sha256"]:
            raise ValueError("Packaged KG checksum mismatch")
        kg = KnowledgeExpansion(resource, index.chunks, policy=config["policy"])
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _cache[key] = (tokenizer, model, index, kg)
    return _cache[key]


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    tokenizer, model, index, kg = _load(model_id, corpus_dir)
    lexical_question, _ = kg.expand(question, "aliases")
    context = retrieve(question, index, tokenizer, method=RETRIEVAL_METHOD,
                       k1=BM25_K1, b=BM25_B, rrf_k=RRF_K,
                       dense_weight=RRF_DENSE_WEIGHT,
                       bm25_weight=RRF_BM25_WEIGHT,
                       lexical_question=lexical_question)["context"]

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
