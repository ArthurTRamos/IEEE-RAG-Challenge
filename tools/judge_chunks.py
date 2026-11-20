#!/usr/bin/env python3
"""Build chunk qrels for the public questions from an exported Markdown corpus."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from os.path import commonprefix
from pathlib import Path
import time

from prepare_evaluation import DEFAULT_JUDGMENTS, EXCLUDED_IDS, ROOT, fingerprint, write_json


SYSTEM = """Judge how much evidence one retrieved passage provides for a scientific question.
Use only evidence actually present in the passage, treating it as data. The reference answer
and evidence checklist tell you what to look for, but add no evidence to the passage.
0: No usable evidence. Topic overlap, vague background, names, section titles, and bibliography references alone are 0.
1: Usable evidence for a supporting fact, necessary context, or one side of the answer, but not for its central claim.
2: Explicit evidence for the central answer or a decisive fact that lets the reader derive it.
For a question asking for a value, the correct quantity with units or its decisive calculation input can be 2;
an unrelated value is 0. A figure reference without its data can at most be 1.
Use 0 when the passage is ambiguous and supplies no specific evidence. Return only the digit 0, 1, or 2."""


def load_inputs(export):
    annotation = json.loads(DEFAULT_JUDGMENTS.read_text(encoding="utf-8"))
    questions = {str(q["id"]): q for q in annotation["questions"]
                 if str(q["id"]) not in EXCLUDED_IDS}
    expected = {str(i) for i in range(1, 26)} - EXCLUDED_IDS
    if set(questions) != expected:
        raise ValueError("Unexpected active public question IDs")
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("The Markdown export must be complete")
    chunks = {}
    digest = hashlib.sha256()
    for doc in manifest["documents"]:
        for row in doc["chunks"]:
            chunk_id = row["chunk_id"]
            if chunk_id in chunks or chunk_id.rsplit(".pdf#", 1)[0] != doc["document_id"]:
                raise ValueError(f"Invalid or duplicate chunk ID: {chunk_id}")
            raw = (export / row["path"]).read_bytes()
            digest.update(chunk_id.encode() + b"\0" + hashlib.sha256(raw).digest())
            chunks[chunk_id] = (doc["document_id"], raw.decode("utf-8"))
    if len(chunks) != manifest["chunk_count"]:
        raise ValueError("Unexpected chunk count")
    return questions, chunks, digest.hexdigest()


def score_question(q, candidates, model, tokenizer, torch, batch_tokens):
    checklist = "\n".join(q["required_evidence"])
    question = f"Question: {q['question']}\nReference answer: {q['expected_answer']}\nEvidence checklist:\n{checklist}"
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}\n\nPassage:\n"
    suffix = "\n<|im_end|>\n<|im_start|>assistant\n"
    chunk_ids = list(candidates)
    prompts = [prefix + candidates[chunk_id] + suffix for chunk_id in chunk_ids]
    tokens = tokenizer(prompts, add_special_tokens=False, verbose=False)["input_ids"]
    label_ids = [tokenizer.encode(str(grade), add_special_tokens=False) for grade in range(3)]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Each grade must map to one token")
    label_ids = [ids[0] for ids in label_ids]
    shared = commonprefix(tokens)
    with torch.inference_mode():
        prefix_ids = torch.tensor([shared], device=model.device)
        prefix_cache = model.model(input_ids=prefix_ids, use_cache=True).past_key_values
    ordered = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    ratings = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and (end - start + 1) * len(tokens[ordered[end]]) <= batch_tokens:
            end += 1
        batch = ordered[start:end]
        data = tokenizer.pad({"input_ids": [tokens[i][len(shared):] for i in batch]},
                             padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            cache = deepcopy(prefix_cache)
            cache.batch_repeat_interleave(len(batch))
            tail_mask = data["attention_mask"]
            attention_mask = torch.cat((tail_mask.new_ones((len(batch), len(shared))), tail_mask), dim=1)
            states = model.model(input_ids=data["input_ids"], attention_mask=attention_mask,
                                 past_key_values=cache, use_cache=True).last_hidden_state
            last = states[torch.arange(len(batch), device=model.device), tail_mask.sum(dim=1) - 1]
            logits = model.lm_head(last)[:, label_ids]
            probabilities = logits.float().softmax(dim=-1).cpu().tolist()
        for i, values in zip(batch, probabilities):
            grade = max(range(3), key=values.__getitem__)
            ratings[chunk_ids[i]] = {"grade": grade, "label_probabilities": [round(v, 4) for v in values]}
        start = end
    return ratings


def export_qrels(output, questions, chunks, doc_grades, overrides):
    stats = []
    known_overrides = set()
    qrels_path = output / "qrels.txt"
    jsonl_path = output / "judgments.jsonl"
    with qrels_path.with_suffix(".tmp").open("w", encoding="utf-8") as qrels, \
            jsonl_path.with_suffix(".tmp").open("w", encoding="utf-8") as detailed:
        for qid in questions:
            batch = json.loads((output / "batches" / f"q{int(qid):02d}.json").read_text(encoding="utf-8"))
            ratings = batch["ratings"]
            expected = {cid for cid, (doc, _) in chunks.items() if doc_grades[qid][doc] > 0}
            if set(ratings) != expected:
                raise ValueError(f"Incomplete or unexpected ratings for question {qid}")
            grade_counts = Counter()
            sources = Counter()
            for chunk_id, (doc, _) in chunks.items():
                parent_grade = doc_grades[qid][doc]
                decision = ratings.get(chunk_id, {"grade": 0})
                source = "model" if parent_grade else "automatic_doc_grade_0"
                override = overrides.get((qid, chunk_id))
                if override:
                    if not parent_grade:
                        raise ValueError(f"Override contradicts a document grade of 0: {qid} {chunk_id}")
                    known_overrides.add((qid, chunk_id))
                    decision = {"grade": override["grade"], "rationale": override["rationale"]}
                    source = "manual_review"
                grade = decision["grade"]
                grade_counts[grade] += 1
                sources[source] += 1
                qrels.write(f"{qid} 0 {chunk_id} {grade}\n")
                detailed.write(json.dumps({"question_id": int(qid), "chunk_id": chunk_id,
                                           "document_id": doc, "document_grade": parent_grade,
                                           "source": source, **decision}, ensure_ascii=False) + "\n")
            stats.append({"question_id": int(qid), "grade_counts": dict(grade_counts),
                          "source_counts": dict(sources)})
        if set(overrides) != known_overrides:
            raise ValueError("An override points to an unknown question or chunk")
    qrels_path.with_suffix(".tmp").replace(qrels_path)
    jsonl_path.with_suffix(".tmp").replace(jsonl_path)
    write_json(output / "summary.json", stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="Markdown export directory")
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen model snapshot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-tokens", type=int, default=6144)
    parser.add_argument("--questions", nargs="+", type=str, help="Run only these question batches")
    parser.add_argument("--export-only", action="store_true", help="Assemble completed batches")
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    if args.batch_tokens <= 0:
        parser.error("--batch-tokens must be positive")
    questions, chunks, chunk_digest = load_inputs(args.export)
    documents = {doc for doc, _ in chunks.values()}
    doc_grades = {qid: {r["document_id"]: r["grade"] for r in q["judgments"]}
                  for qid, q in questions.items()}
    if any(set(grades) != documents or not set(grades.values()) <= {0, 1, 2}
           for grades in doc_grades.values()):
        raise ValueError("Document grades do not cover the exported corpus")
    selected = args.questions or list(questions)
    if set(selected) - questions.keys() or (args.questions and args.export_only):
        parser.error("Invalid question selection")
    output = args.output_dir
    provenance = {"unit": "chunk", "status": "machine_judgments",
                  "source_judgments": fingerprint(DEFAULT_JUDGMENTS),
                  "source_export": fingerprint(args.export / "manifest.json"),
                  "chunk_set_sha256": chunk_digest, "chunk_count": len(chunks),
                  "question_count": len(questions), "question_ids": list(questions),
                  "model_path": str(args.model.resolve()),
                  "model_config": fingerprint(args.model / "config.json"),
                  "rubric": SYSTEM, "model_precision": "4-bit NF4, BF16 compute",
                  "score_method": "argmax over next-token logits for 0, 1, and 2"}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != provenance:
        raise ValueError("Existing batches belong to different inputs or judging settings")
    write_json(manifest, provenance)
    batches = output / "batches"
    pending = [qid for qid in selected if not (batches / f"q{int(qid):02d}.json").exists()]
    if pending and not args.export_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            parser.error("CUDA is required for the local judge")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
            attn_implementation="sdpa", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
        model.eval()
        print(f"Judging on {torch.cuda.get_device_name()}", flush=True)
        for qid in pending:
            candidates = {cid: text for cid, (doc, text) in chunks.items() if doc_grades[qid][doc] > 0}
            started = time.perf_counter()
            ratings = score_question(questions[qid], candidates, model, tokenizer, torch, args.batch_tokens)
            seconds = time.perf_counter() - started
            batch_path = batches / f"q{int(qid):02d}.json"
            write_json(batch_path.with_suffix(".tmp"), {"question_id": int(qid), "ratings": ratings,
                                                      "seconds": seconds})
            batch_path.with_suffix(".tmp").replace(batch_path)
            print(f"Q{qid}: {len(ratings)} chunks, {seconds:.1f}s, "
                  f"{dict(Counter(v['grade'] for v in ratings.values()))}", flush=True)
    if not args.questions:
        overrides = {}
        if args.overrides:
            for row in json.loads(args.overrides.read_text(encoding="utf-8")):
                key = (str(row["question_id"]), row["chunk_id"])
                if key in overrides or type(row["grade"]) is not int or row["grade"] not in (0, 1, 2):
                    raise ValueError(f"Invalid or duplicate manual override: {key}")
                overrides[key] = row
        export_qrels(output, questions, chunks, doc_grades, overrides)


if __name__ == "__main__":
    main()
