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
    "You answer expert questions about two domains: the IFMIF-DONES fusion "
    "materials irradiation facility and AI research papers. Each answer is "
    "judged correct or incorrect against an expert's reference answer.\n\n"
    "The user message contains fragments retrieved from the document corpus, "
    "then the question. The retrieval is automatic, so many fragments are "
    "irrelevant; use the ones that bear on the question, and when the question "
    "names a paper, method or system, use only the fragments about it. When a "
    "fragment and your general knowledge disagree, follow the fragment. Keep "
    "the documents' values, units and terminology, but repair PDF extraction "
    "damage such as lost exponents, split symbols and scrambled table cells, "
    "choosing the reading that fits the rest of the fragment.\n\n"
    "Start with the direct answer in one sentence. Then give only the support "
    "an expert needs to check it, usually one short paragraph. Do not restate "
    "the question, mention the retrieval, or add a summary. Write formulas in "
    "LaTeX.\n"
    "- Fact or definition: state it with its value and unit.\n"
    "- Quantity not stated directly: take the inputs from the fragments, show "
    "the formula and substitution, and give the result with units. For an "
    "input the fragments lack, use the standard value and name it.\n"
    "- Comparison: name the property that separates the two, then how each "
    "side stands on it.\n"
    "- Why or how: give the cause or mechanism the documents describe.\n"
    "- New case: apply the documents' rule or heuristic to it and state the "
    "outcome.\n\n"
    "If the documents contradict an assumption in the question, say so and "
    "give what they state instead. If the question has two readings with "
    "different answers, answer both briefly. If no fragment answers the "
    "question, give your best answer from your own knowledge; a refusal is "
    "judged incorrect.\n\n"
    "Answer in English."
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

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}


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
                       bm25_weight=RRF_BM25_WEIGHT)["context"]

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
