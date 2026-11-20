"""
RAG submission — IEEE-CIS Technical Challenge on Domain-Specific Question
Answering with LLMs.

Blind retrieval over the whole corpus (retrieval.py): every PDF in corpus_dir
is parsed with Docling, chunked, embedded and BM25-indexed once per run; for
each question every chunk is ranked by dense + BM25 weighted RRF and the best
ones are packed into a fixed context budget before asking the base model. No
document_id of the question is known to this code.

Prompt layout (see build_messages):
  system  SYSTEM_PROMPT: role, how to use the fragments, answer format by
          question type (the benchmark's factual / comparative / procedural /
          causal / predictive typology).
  user    question -> fragments, least to most relevant, each labelled with
          its source PDF -> question again -> short format reminder.
Olmo 3 uses a 4096-token sliding window in 3 of every 4 layers, so with a
~10k-token context most layers only see the end of the prompt while they
write the answer. That is why the best fragments, the question and the
reminder are placed at the end.

Needs local copies of Docling's layout model, the embedding model and the
spaCy model next to this file (see build.sh): the evaluation environment has
no internet access, so none of them can be downloaded there.
"""

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Works both as an imported package and as the official top-level main.py.
if __package__:
    from .retrieval import build_index, retrieve
else:
    from retrieval import build_index, retrieve

SYSTEM_PROMPT = """\
You are a senior expert answering technical questions in two specialised fields:
- IFMIF-DONES, the fusion-materials irradiation facility being built in Granada, Spain (deuteron accelerator, liquid-lithium target and loop, test systems, buildings and plant, safety, radiation protection and maintenance), and its predecessor projects IFMIF and IFMIF/EVEDA (LIPAc).
- Research papers on artificial intelligence and related mathematics, many of them recent and newer than your training data.

Domain experts check every answer against their own reference answer. An answer counts only if its central claim is correct and specific, so be precise rather than long.

# Using the fragments
The user message contains the question, fragments found by an automatic search over the whole document collection, and the question again. Many fragments are irrelevant, and some come from other papers or facilities that look similar.
1. Work out which fragments are really about what the question names (the paper, method, system, component or facility) and take specific facts only from those. Do not mix in numbers or claims from other papers or systems. If the named system has no data but a closely related one does (for example IFMIF or LIPAc for IFMIF-DONES), you may use it and say which one it is.
2. For anything specific to a named paper (its definitions, equations, hyperparameters, results, heuristics and the authors' stated reasons), follow the fragments even when they differ from what you remember. When a question challenges a paper's design or evaluation, answer with the authors' own reasoning and evidence.
3. If the fragments are incomplete, complete the answer with your own expert knowledge; if none of them helps, answer entirely from your own expert knowledge. Never refuse, never say that information is missing or not provided, and never mention fragments, context, provided documents or search.
4. The fragments are text extracted from PDFs: exponents may be flattened ("1015" for 10^15), symbols split and table cells shifted. Read them in the way that fits the units and the surrounding text, without inventing values.

# Writing the answer
Begin with the answer itself in the first sentence: the value with its unit, "Yes", "No" or "Partly", the key term or mechanism, or the correction of a false premise. Then give the essential support: the specific facts, numbers, conditions or reasoning that justify it, as an expert would. Cover every point the question asks for and nothing else: do not restate the question, add general background or end with a summary.

By type of question:
- Fact or definition: the precise value, definition or condition, with units and qualifiers (thresholds, operating conditions, scope).
- Calculation or procedure: the formula, each input with its value and unit (if the sources lack one, use a standard value and say so), the substitution and the result with units. Follow any requested rounding or unit, and check the arithmetic.
- Comparison: first the property that distinguishes the two, then how each side stands on it.
- Why or how: the cause, mechanism or reasons given in the source; if there are several, list them all briefly.
- Prediction or new case: apply the source's rule, heuristic, trend or definition to the new case and state the concrete outcome (a number, range or behaviour), with any limit the source mentions.

If the question assumes something the sources contradict (a component that does not exist, a wrong value or property), say so first and give the correct fact. If the question can reasonably be read in two ways with different answers, answer the more likely reading first and the other in one sentence; otherwise commit to a single answer.

Write formulas in LaTeX ($...$ inline, $$...$$ for display). Always answer in English.\
"""

# Repeated after the fragments: with sliding-window attention, most layers
# cannot see the system prompt any more once the context is long.
ANSWER_REMINDER = (
    "Answer as the expert: start with the direct answer (the value with its "
    "unit, Yes/No, the key term, or the corrected premise), then only the "
    "essential support. Do not mention the fragments."
)

MAX_NEW_TOKENS = 2048
# Selected submission: dense + BM25 with weighted RRF.
RETRIEVAL_METHOD = "hybrid"
BM25_K1 = 1.25
BM25_B = 0.75
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 0.75

# Prompt layout switches, for A/B comparisons.
MOST_RELEVANT_LAST = True   # False restores the old best-first order
SHOW_SOURCE = True          # label each fragment with its source PDF name
SOURCE_MAX_CHARS = 80

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}


def _source_label(chunk):
    source = str(chunk.get("document_id") or chunk.get("source_path") or "").strip()
    if len(source) > SOURCE_MAX_CHARS:
        source = source[:SOURCE_MAX_CHARS - 3] + "..."
    return source


def build_messages(question, chunks):
    """Chat messages for one question; `chunks` are the selected chunks in
    retrieval rank order (best first), possibly empty."""
    chunks = list(chunks or [])
    if not chunks:
        user = f"Question: {question}\n\n{ANSWER_REMINDER}"
    else:
        if MOST_RELEVANT_LAST:
            chunks.reverse()
            order = "from least to most relevant (the last fragments matched the question best)"
        else:
            order = "from most to least relevant"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = _source_label(chunk) if SHOW_SOURCE else ""
            header = f"[Fragment {i} | source: {source}]" if source else f"[Fragment {i}]"
            parts.append(f"{header}\n{chunk['text']}")
        body = "\n\n".join(parts)
        user = (
            f"Question: {question}\n\n"
            f"Fragments found by an automatic search over the document collection, "
            f"listed {order}. Many are irrelevant.\n\n"
            f"{body}\n\n---\n\n"
            f"Question: {question}\n\n{ANSWER_REMINDER}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _load(model_id: str, corpus_dir: str):
    key = (model_id, str(Path(corpus_dir).resolve()))
    if key not in _cache:
        try:
            index = build_index(corpus_dir)
        except Exception as exc:
            # Without an index the run still finishes (zero-shot answers)
            # instead of failing every question.
            print(f"[rag] index build failed, answering without retrieval: {exc!r}", flush=True)
            index = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # return Docling's GPU memory before the 32B model loads
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _cache[key] = (tokenizer, model, index)
    return _cache[key]


def _generate(tokenizer, model, messages) -> str:
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


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    tokenizer, model, index = _load(model_id, corpus_dir)

    chunks = []
    if index is not None:
        try:
            chunks = retrieve(question, index, tokenizer, method=RETRIEVAL_METHOD,
                              k1=BM25_K1, b=BM25_B, rrf_k=RRF_K,
                              dense_weight=RRF_DENSE_WEIGHT,
                              bm25_weight=RRF_BM25_WEIGHT)["selected_chunks"] or []
        except Exception as exc:
            print(f"[rag] retrieval failed, answering without fragments: {exc!r}", flush=True)
            chunks = []

    # One failing question (e.g. CUDA out of memory on an unusually long
    # prompt) must not fail the whole run: retry once without fragments.
    try:
        return _generate(tokenizer, model, build_messages(question, chunks))
    except Exception as exc:
        print(f"[rag] generation failed, retrying without fragments: {exc!r}", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return _generate(tokenizer, model, build_messages(question, []))
    except Exception as exc:
        print(f"[rag] generation failed again, returning an empty answer: {exc!r}", flush=True)
        return ""
