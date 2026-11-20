#!/usr/bin/env python3
"""Evaluate src's current RRF/context on the hash-verified, judged Markdown export.

No generative weights or inference. Embeddings and lexical scores are recomputed;
PDF conversion is reused so chunk judgments refer to exactly the evaluated text.
"""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from prepare_evaluation import fingerprint, prepare, write_json


def submission_settings(path=ROOT / "src/main.py"):
    """Read literal settings without importing main or its generative loader."""
    names = {"RETRIEVAL_METHOD", "BM25_K1", "BM25_B", "RRF_K",
             "RRF_DENSE_WEIGHT", "RRF_BM25_WEIGHT", "SYSTEM_PROMPT"}
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if values.keys() != names:
        raise ValueError("Submission settings must be explicit literal assignments")
    return values


def context_metrics(selected_ids, grades):
    """Graded nDCG; binary RR/recall at >=1 and >=2, using the ENTIRE qrels.

    Each chunk occupies its own rank, including chunks from the same document.
    Queries with no positives have undefined RR/recall (None), not perfect recall.
    """
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate chunk in context")
    if set(selected_ids) - grades.keys():
        raise ValueError("Unjudged chunk in context")
    if any(g not in (0, 1, 2) for g in grades.values()):
        raise ValueError("Chunk grades must be 0, 1 or 2")
    observed = [grades[cid] for cid in selected_ids]
    ideal = sorted(grades.values(), reverse=True)[:len(observed)]

    def dcg(values):
        return sum((2 ** g - 1) / math.log2(rank + 1)
                   for rank, g in enumerate(values, 1))

    idcg = dcg(ideal)
    result = {"k": len(observed), "dcg": dcg(observed), "idcg": idcg,
              "ndcg": dcg(observed) / idcg if idcg else
              (0.0 if any(grades.values()) else None),
              "grade_counts": {str(g): observed.count(g) for g in (0, 1, 2)}}
    for threshold in (1, 2):
        total = sum(g >= threshold for g in grades.values())
        positions = [i for i, g in enumerate(observed, 1) if g >= threshold]
        result[f"ge{threshold}"] = {
            "available": total, "retrieved": len(positions),
            "recall": len(positions) / total if total else None,
            "reciprocal_rank": (1 / positions[0] if positions else 0.0) if total else None,
            "precision": len(positions) / len(observed) if observed else 0.0,
            "hit": int(bool(positions)) if total else None,
        }
    return result


def document_diversity(selected):
    counts = Counter(c["document_id"] for c in selected)
    tokens = Counter()
    for chunk in selected:
        tokens[chunk["document_id"]] += chunk["tokens"]
    total, total_tokens = len(selected), sum(tokens.values())
    shares = [n / total for n in counts.values()]
    entropy = -sum(p * math.log(p) for p in shares)
    return {
        "unique_documents": len(counts), "chunks_per_document": dict(counts.most_common()),
        "tokens_per_document": dict(tokens.most_common()),
        "dominant_document_chunk_share": max(shares, default=0.0),
        "dominant_document_token_share": max(tokens.values(), default=0) / total_tokens
        if total_tokens else 0.0,
        "repeated_document_chunk_share": 1 - len(counts) / total if total else 0.0,
        "shannon_entropy_nats": entropy,
        "effective_documents": math.exp(entropy) if total else 0.0,
    }


def prompt_token_count(tokenizer, messages):
    # transformers 5 can return a BatchEncoding by default. Its len() counts
    # fields, not tokens. Use the same explicit tensor/dict API as src/main.py.
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
    return encoded["input_ids"].shape[-1]


def channel_top_chunks(ranking, canonical_chunks, limit=100):
    """Recover native channel orders from this run's scores, before RRF.

    Canonical corpus order and original score dtypes preserve src's tie rules:
    NumPy default argsort for dense, stable argsort for BM25. Do not sort the
    fused list directly: its tie order is different from the input corpus.
    Raw BM25 top-k includes zero/negative scores; RRF only counts positive ones.
    """
    import numpy as np

    by_id = {row["chunk_id"]: row for row in ranking}
    if len(by_id) != len(ranking) or set(by_id) != {c["chunk_id"] for c in canonical_chunks}:
        raise ValueError("Channel scores must cover the canonical corpus exactly")
    result = {}
    for channel, field, dtype in (("semantic", "dense_score", np.float32),
                                  ("bm25", "bm25_score", np.float64)):
        scores = np.asarray([by_id[c["chunk_id"]][field] for c in canonical_chunks], dtype=dtype)
        order = np.argsort(-scores, **({"kind": "stable"} if channel == "bm25" else {}))
        result[channel] = []
        for rank, i in enumerate(order[:limit], 1):
            row = by_id[canonical_chunks[int(i)]["chunk_id"]]
            result[channel].append({**row, "rank": rank, "score": float(scores[i]),
                                    "rrf_rank": row["rank"], "rrf_score": row["score"]})
    return result


def save_channel_top100(output, question, ranking, chunks, judgments, tokens, selected_ids):
    stem = f"{int(question['id']):02}"
    for channel, top in channel_top_chunks(ranking, chunks).items():
        annotated = [{**c, "tokens": tokens[c["chunk_id"]],
                      "in_context": c["chunk_id"] in selected_ids,
                      **{key: judgments[c["chunk_id"]][key]
                         for key in ("grade", "document_grade", "rationale")}}
                     for c in top]
        folder = output / "top100" / channel
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / f"{stem}.json", {
            "id": str(question["id"]), "question": question["question"],
            "channel": channel, "score_type": "cosine_similarity" if channel == "semantic" else "BM25Okapi",
            "stage": "before_rrf", "requested_k": 100, "returned_k": len(annotated),
            "chunks": annotated,
        })
        lines = [f"# Pergunta {question['id']} — top-100 {channel}", "", question["question"], "",
                 "Ranking individual antes do RRF; score é o score original deste canal. "
                 "O RRF continua usando o corpus inteiro, sem corte em 100 candidatos. "
                 "BM25 inclui scores não positivos se necessário para completar 100 resultados, "
                 "embora somente scores positivos contribuam para o RRF.", ""]
        for chunk in annotated:
            lines += [f"## {chunk['rank']}. {chunk['chunk_id']}", "",
                      f"Score: {chunk['score']:.10g} | Rank RRF: {chunk['rrf_rank']} | "
                      f"Grau chunk: {chunk['grade']} | Grau documento: {chunk['document_grade']} | "
                      f"Tokens: {chunk['tokens']} | No contexto: {'sim' if chunk['in_context'] else 'não'}", "",
                      chunk["text"], "", f"Justificativa: {chunk['rationale']}", ""]
        (folder / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def distribution(values):
    return {"mean": statistics.mean(values), "stddev": statistics.pstdev(values),
            "min": min(values), "median": statistics.median(values), "max": max(values)}


def aggregate(rows):
    result = {"questions": len(rows)}
    for scope in ("context", "context_top10"):
        result[scope] = {}
        for name, key in (("ndcg", None), ("mrr_ge1", ("ge1", "reciprocal_rank")),
                          ("mrr_ge2", ("ge2", "reciprocal_rank")),
                          ("recall_ge1", ("ge1", "recall")),
                          ("recall_ge2", ("ge2", "recall")),
                          ("precision_ge1", ("ge1", "precision")),
                          ("precision_ge2", ("ge2", "precision")),
                          ("hit_rate_ge1", ("ge1", "hit")),
                          ("hit_rate_ge2", ("ge2", "hit"))):
            pairs = [(q["id"], q[scope][key[0]][key[1]] if key else q[scope]["ndcg"])
                     for q in rows]
            defined = [v for _, v in pairs if v is not None]
            result[scope][name] = {
                "mean": statistics.mean(defined) if defined else None,
                "defined_queries": len(defined),
                "undefined_question_ids": [qid for qid, v in pairs if v is None],
            }
    result["distributions"] = {
        "context_chunks": distribution([q["context"]["k"] for q in rows]),
        "chunk_tokens": distribution([q["chunk_tokens"] for q in rows]),
        "prompt_tokens": distribution([q["prompt_tokens"] for q in rows]),
        **{name: distribution([q["diversity"][name] for q in rows]) for name in (
            "unique_documents", "dominant_document_chunk_share",
            "dominant_document_token_share", "effective_documents")},
    }
    result["grade_counts"] = dict(sum((Counter(q["context"]["grade_counts"]) for q in rows), Counter()))
    result["documents_seen"] = dict(Counter(
        doc for q in rows for doc in q["diversity"]["chunks_per_document"]).most_common())
    result["unique_documents_across_questions"] = len(result["documents_seen"])
    result["micro_recall_ge1"] = sum(q["context"]["ge1"]["retrieved"] for q in rows) / sum(
        q["context"]["ge1"]["available"] for q in rows)
    result["micro_recall_ge2"] = sum(q["context"]["ge2"]["retrieved"] for q in rows) / sum(
        q["context"]["ge2"]["available"] for q in rows)
    return result


def load_inputs(export_dir, judgments_dir, corpus):
    from src.retrieval import _chunk_markdown

    export_path = export_dir / "manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    annotation = json.loads((judgments_dir / "manifest.json").read_text(encoding="utf-8"))
    if annotation["status"] != "complete" or annotation["source_export"]["sha256"] != fingerprint(export_path)["sha256"]:
        raise ValueError("Incomplete annotations or different source export")
    for source in annotation["review_sources"] + [annotation["source_judgments"]]:
        if fingerprint(source["path"])["sha256"] != source["sha256"]:
            raise ValueError(f"Annotation source changed: {source['path']}")
    questions, _, corpus_manifest = prepare(corpus)
    documents = sorted(export["documents"], key=lambda d: d["source_path"])
    if {d["document_id"] for d in documents} != {
            p.stem for p in corpus.rglob("*.pdf")}:
        raise ValueError("Export and corpus contain different documents")
    chunks = []
    for doc in documents:
        md = export_dir / doc["markdown_path"]
        pdf = corpus / doc["source_path"]
        if fingerprint(md)["sha256"] != doc["markdown_sha256"] or fingerprint(pdf)["sha256"] != doc["source_pdf"]["sha256"]:
            raise ValueError(f"Document hash changed: {doc['document_id']}")
        raw_text = md.read_text(encoding="utf-8")
        if "index_markdown_path" in doc:
            from src.retrieval import clean_markdown
            index_md = export_dir / doc["index_markdown_path"]
            if fingerprint(index_md)["sha256"] != doc["index_markdown_sha256"]:
                raise ValueError(f"Index Markdown hash changed: {doc['document_id']}")
            index_text = index_md.read_text(encoding="utf-8")
            if clean_markdown(raw_text) != index_text:
                raise ValueError("Current cleanup differs from indexed export")
        else:
            # Historical exports remain reproducible against their original qrels.
            index_text = raw_text
        parts = _chunk_markdown(index_text)
        if len(parts) != len(doc["chunks"]):
            raise ValueError("Current chunking differs from judged export")
        for number, (part, saved) in enumerate(zip(parts, doc["chunks"])):
            path = export_dir / saved["path"]
            cid = f"{doc['document_id']}.pdf#{number}"
            if saved["chunk_id"] != cid or fingerprint(path)["sha256"] != saved["sha256"] or path.read_text(encoding="utf-8") != part:
                raise ValueError(f"Chunk changed: {cid}")
            chunks.append({"chunk_id": cid, "document_id": doc["document_id"],
                           "source_path": doc["source_path"], "text": part})
    ids = {c["chunk_id"] for c in chunks}
    qrels = {}
    for q in questions:
        qid = str(q["id"])
        payload = json.loads((judgments_dir / "questions" / f"{int(qid):02}.json").read_text(encoding="utf-8"))
        judged = {r["chunk_id"]: r for r in payload["judgments"]}
        if payload["question"] != q["question"] or set(judged) != ids or len(judged) != len(payload["judgments"]):
            raise ValueError(f"Judgment coverage mismatch: {qid}")
        for chunk in chunks:
            r = judged[chunk["chunk_id"]]
            if r["document_id"] != chunk["document_id"] or r["grade"] not in (0, 1, 2) or (r["document_grade"] == 0 and r["grade"] != 0):
                raise ValueError(f"Invalid annotation: {qid}/{chunk['chunk_id']}")
        qrels[qid] = judged
    return questions, chunks, qrels, corpus_manifest, export


def write_report(output, rows, summary, manifest):
    def fmt(x):
        return f"{x:.4f}" if x is not None else "—"

    cfg = manifest["settings"]
    lines = ["# RRF e contexto: avaliação por chunk", "",
             f"Execução: {manifest['created_at']}. {manifest['document_count']} documentos, "
             f"{manifest['chunk_count']} chunks, {len(rows)} perguntas. Nenhuma inferência generativa.", "",
             f"Configuração lida de `src/main.py`: {cfg['RETRIEVAL_METHOD']}, "
             f"BM25 k1={cfg['BM25_K1']} / b={cfg['BM25_B']}; RRF k={cfg['RRF_K']}, "
             f"peso denso={cfg['RRF_DENSE_WEIGHT']} / BM25={cfg['RRF_BM25_WEIGHT']}. "
             "BM25 atual usa lemas spaCy, remoção "
             "de stopwords e normalização Unicode. `src.retrieval.retrieve` e sua seleção "
             "original de contexto foram chamados diretamente.", "",
             "Embeddings e rankings recalculados sobre os Markdown já exportados. Hashes de "
             "PDFs, Markdown e chunks verificados, e chunking reproduzido com o código atual. "
             "AI-029 inclui a recuperação da extração anterior, registrada no manifesto de origem. "
             "O tempo desta execução não inclui conversão dos PDFs.", "",
             "## Métricas macro", "",
             "nDCG usa ganho `2^grau - 1`, desconto `log2(rank + 1)` e ranking ideal dos "
             f"{manifest['chunk_count']} chunks julgados para a pergunta. K é o número de chunks efetivamente "
             "selecionados no orçamento de 10.000 tokens. Chunks do mesmo documento ocupam "
             "posições distintas. A coluna top 10 considera apenas os primeiros 10 chunks do contexto.", "",
             "| Métrica | Contexto completo | Top 10 do contexto |", "|---|---:|---:|"]
    for name in ("ndcg", "mrr_ge2", "recall_ge2", "mrr_ge1", "recall_ge1", "precision_ge1", "hit_rate_ge2"):
        lines.append(f"| {name} | {fmt(summary['context'][name]['mean'])} | {fmt(summary['context_top10'][name]['mean'])} |")
    lines += ["", "`ge2`: grau 2. `ge1`: graus 1 ou 2. Médias por pergunta, sem ponderar "
              "pelo número de positivos. A pergunta 4 não possui chunks de grau 2: seu RR/recall "
              "ge2 é indefinido e excluído dessas médias (21 perguntas); nDCG usa as 22 perguntas. "
              "Essas notas foram produzidas por revisão Codex, sem validação humana independente, "
              "com zeros de documentos herdados. Métricas não medem acurácia de respostas.", "",
              "## Resultados por pergunta", "",
              "| ID | Chunks | Docs | Maior parcela por doc | Tokens chunks | nDCG@K | RR ge2 | Recall ge2 | Recall ge1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in rows:
        m, d = q["context"], q["diversity"]
        lines.append(f"| [{q['id']}](questions/{int(q['id']):02}.md) | {m['k']} | {d['unique_documents']} | "
                     f"{d['dominant_document_chunk_share']:.1%} | {q['chunk_tokens']} | {fmt(m['ndcg'])} | "
                     f"{fmt(m['ge2']['reciprocal_rank'])} | {fmt(m['ge2']['recall'])} | {fmt(m['ge1']['recall'])} |")
    lines += ["", "## Variabilidade documental e recursos", ""]
    for key in ("context_chunks", "unique_documents", "dominant_document_chunk_share", "effective_documents", "prompt_tokens"):
        d = summary["distributions"][key]
        lines.append(f"- {key}: média {d['mean']:.3f}, desvio {d['stddev']:.3f}, mínimo {d['min']:.3f}, máximo {d['max']:.3f}.")
    lines += [f"- Documentos distintos na união dos contextos: {summary['unique_documents_across_questions']}.",
              f"- Contagens de grau no contexto, somando perguntas: {summary['grade_counts']}.",
              f"- GPU: {manifest['gpu']}; pico de memória alocada PyTorch: {manifest['torch_peak_gpu_allocated_bytes'] / 2**20:.1f} MiB.",
              f"- Tempo total: {manifest['elapsed_seconds']:.2f} s; decomposição em `manifest.json`.", "",
              "A diversidade é descritiva: muitos documentos não garantem boas evidências. "
              "Documentos efetivos = exponencial da entropia das proporções de chunks por documento; "
              "vale 1 quando todos os chunks vêm do mesmo documento.", "",
              "O seletor para no primeiro chunk que excede o orçamento e não tenta preencher "
              "o saldo com chunks posteriores. Os 10.000 tokens contam apenas os textos dos chunks, "
              "sem cabeçalhos, pergunta, sistema e template de chat. `prompt_tokens` mede esse prompt "
              "completo usando somente o tokenizer local do OLMo.", "",
              "## Artefatos e limites", "",
              "`contexts/NN.md`: contexto exato que seria passado ao OLMo; `questions/NN.json`: "
              "textos, IDs, notas, justificativas, scores e tokens; `rankings/NN.json`: RRF completo; "
              "`top100/semantic/NN.json` e `.md`: top-100 semântico antes do RRF; "
              "`top100/bm25/NN.json` e `.md`: top-100 BM25 antes do RRF. Esses arquivos "
              "incluem texto integral, score do canal, posição no RRF, nota de relevância e "
              "indicação de inclusão no contexto. O RRF usa todos os candidatos originais; "
              "o corte em 100 serve apenas para exportação. "
              "`summary.csv`: métricas e composição por pergunta; `metrics.json`: resultados completos. "
              "As notas e respostas de referência não participam da recuperação.", "",
              "Não foram alterados `src`, a interface oficial, as dependências ou o pacote de submissão. "
              "Execução local na RTX 4050, sem inferência: não estabelece o tempo total de 60 minutos "
              "nem acurácia do sistema na H100 ou no corpus oculto."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=ROOT / "experiments/runs/pdf-markdown-chunks")
    parser.add_argument("--judgments", type=Path, default=ROOT / "experiments/judments/public22-chunks")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/olmo-tokenizer")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    started = time.perf_counter()
    settings = submission_settings()
    questions, chunks, qrels, corpus_manifest, export = load_inputs(args.export, args.judgments, args.corpus)
    validation_seconds = time.perf_counter() - started
    print(f"Validated {len(chunks)} chunks / {len(questions)} questions", flush=True)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from src.retrieval import EMBED_MODEL_PATH, MAX_CONTEXT_TOKENS, BM25Retriever, RetrievalIndex, retrieve

    if not torch.cuda.is_available():
        raise RuntimeError("Dedicated GPU unavailable; run with GPU access")
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    # Match build_index's device choice, batch size, normalization and precision.
    embedder = SentenceTransformer(str(EMBED_MODEL_PATH), device=None, local_files_only=True)
    if embedder.device.type != "cuda":
        raise RuntimeError("Embedding model did not select CUDA")
    embeddings = np.asarray(embedder.encode([c["text"] for c in chunks],
        normalize_embeddings=True, show_progress_bar=False, batch_size=64), dtype=np.float32)
    torch.cuda.synchronize()
    embedding_seconds = time.perf_counter() - t
    print(f"Fresh embeddings on {embedder.device}: {embedding_seconds:.2f}s", flush=True)
    t = time.perf_counter()
    lexical = BM25Retriever(chunks, settings["BM25_K1"], settings["BM25_B"])
    lexical_seconds = time.perf_counter() - t
    t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    tokens = {c["chunk_id"]: len(tokenizer.encode(c["text"])) for c in chunks}
    tokenizer_seconds = time.perf_counter() - t
    index = RetrievalIndex(chunks, embeddings, embedder, sorted({c["document_id"] for c in chunks}), [], 0., lexical)
    for subdir in ("contexts", "questions", "rankings"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)
    write_json(args.output_dir / "chunk-order.json", [c["chunk_id"] for c in chunks])
    rows, retrieval_seconds = [], 0.0
    for question in questions:
        qid, text = str(question["id"]), question["question"]
        t = time.perf_counter()
        result = retrieve(text, index, tokenizer, method=settings["RETRIEVAL_METHOD"],
            k1=settings["BM25_K1"], b=settings["BM25_B"], rrf_k=settings["RRF_K"],
            dense_weight=settings["RRF_DENSE_WEIGHT"], bm25_weight=settings["RRF_BM25_WEIGHT"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        retrieval_seconds += seconds
        selected = [{**c, "tokens": len(tokenizer.encode(c["text"])),
                     **{k: qrels[qid][c["chunk_id"]][k] for k in ("grade", "document_grade", "rationale")}}
                    for c in result["selected_chunks"]]
        if any(c["truncated"] for c in selected):
            raise ValueError(f"Question {qid}: truncated chunk needs its own semantic judgment")
        ids = [c["chunk_id"] for c in selected]
        grades = {cid: r["grade"] for cid, r in qrels[qid].items()}
        prompt = [{"role": "system", "content": settings["SYSTEM_PROMPT"]},
                  {"role": "user", "content": result["context"] + text}]
        excluded = result["chunks"][len(selected)] if len(selected) < len(chunks) else None
        row = {"id": qid, "question": text, "domain": question["domain"],
               "context": context_metrics(ids, grades),
               "context_top10": context_metrics(ids[:10], grades),
               "diversity": document_diversity(selected), "retrieval_seconds": seconds,
               "chunk_tokens": sum(c["tokens"] for c in selected),
               "prompt_tokens": prompt_token_count(tokenizer, prompt), "first_excluded_chunk":
               {"chunk_id": excluded["chunk_id"], "tokens": tokens[excluded["chunk_id"]],
                "grade": grades[excluded["chunk_id"]]} if excluded else None,
               "irrelevant_chunk_tokens": sum(c["tokens"] for c in selected if c["grade"] == 0)}
        rows.append(row)
        stem = f"{int(qid):02}"
        save_channel_top100(args.output_dir, question, result["chunks"], chunks,
                            qrels[qid], tokens, set(ids))
        (args.output_dir / "contexts" / f"{stem}.md").write_text(result["context"], encoding="utf-8")
        write_json(args.output_dir / "questions" / f"{stem}.json", {**row, "selected_chunks": selected})
        compact = [{k: c[k] for k in ("chunk_id", "document_id", "rank", "score", "dense_score", "bm25_score")}
                   for c in result["chunks"]]
        write_json(args.output_dir / "rankings" / f"{stem}.json", {"id": qid, "chunks": compact})
        audit = [f"# Pergunta {qid}", "", text, "",
                 f"[Contexto exato](../contexts/{stem}.md) · [JSON com textos e notas]({stem}.json)", "",
                 f"[Top-100 semântico](../top100/semantic/{stem}.md) · [Top-100 BM25](../top100/bm25/{stem}.md)", "",
                 "| Rank | Chunk | Grau chunk | Grau doc | Tokens | RRF |", "|---:|---|---:|---:|---:|---:|"]
        for c in selected:
            audit.append(f"| {c['rank']} | {c['chunk_id']} | {c['grade']} | {c['document_grade']} | {c['tokens']} | {c['score']:.6f} |")
        (args.output_dir / "questions" / f"{stem}.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
        print(f"Q{qid}: K={len(ids)} docs={row['diversity']['unique_documents']} nDCG={row['context']['ndcg']:.4f}", flush=True)

    summary = aggregate(rows)
    write_json(args.output_dir / "metrics.json", {"summary": summary, "questions": rows,
        "by_domain": {domain: aggregate([q for q in rows if q["domain"] == domain])
                      for domain in sorted({q["domain"] for q in rows})}})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "domain", "chunks", "documents", "chunk_tokens", "prompt_tokens", "ndcg_context",
                         "ndcg_top10", "rr_ge2", "recall_ge2", "rr_ge1", "recall_ge1", "dominant_doc_share", "chunks_per_document"])
        for q in rows:
            m, d = q["context"], q["diversity"]
            writer.writerow([q["id"], q["domain"], m["k"], d["unique_documents"], q["chunk_tokens"], q["prompt_tokens"],
                m["ndcg"], q["context_top10"]["ndcg"], m["ge2"]["reciprocal_rank"], m["ge2"]["recall"],
                m["ge1"]["reciprocal_rank"], m["ge1"]["recall"], d["dominant_document_chunk_share"],
                json.dumps(d["chunks_per_document"])])
    manifest = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {k: v for k, v in settings.items() if k != "SYSTEM_PROMPT"},
        "max_context_tokens": MAX_CONTEXT_TOKENS, "embedding_model": str(EMBED_MODEL_PATH.resolve()),
        "embedding_max_sequence_length": embedder.max_seq_length,
        "embedding_dimensions": embeddings.shape[1], "embedding_batch_size": 64,
        "tokenizer": str(args.tokenizer.resolve()), "gpu": torch.cuda.get_device_name(),
        "generative_model_loaded": False, "generative_inference": False,
        "fresh_embeddings": True, "fresh_bm25": True, "pdf_conversion_reused": True,
        "channel_top_k_exported": 100,
        "channel_top_k_is_rrf_candidate_limit": False,
        "channel_export_formats": ["json", "markdown_with_full_chunk_text"],
        "document_count": len(index.documents), "chunk_count": len(chunks), "question_count": len(questions),
        "validation_seconds": validation_seconds, "embedding_seconds": embedding_seconds,
        "lexical_index_seconds": lexical_seconds, "tokenizer_and_counts_seconds": tokenizer_seconds,
        "retrieval_seconds": retrieval_seconds, "elapsed_seconds": time.perf_counter() - started,
        "torch_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "versions": {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy", "spacy", "rank-bm25")},
        "code": [fingerprint(p) for p in (Path(__file__), ROOT / "src/main.py", ROOT / "src/retrieval.py")],
        "sources": [fingerprint(args.export / "manifest.json"), fingerprint(args.judgments / "manifest.json")],
        "annotation_files": [fingerprint(args.judgments / "questions" / f"{int(q['id']):02}.json") for q in rows],
        "model_files": [fingerprint(p) for folder in (EMBED_MODEL_PATH, args.tokenizer, ROOT / "src/spacy_model")
                        for p in sorted(folder.rglob("*")) if p.is_file() and ".cache" not in p.parts],
        "recovered_extraction_failures": export.get("recovered_failures", []),
        "corpus_coverage": corpus_manifest,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir, rows, summary, manifest)
    index_lines = ["# Top-100 por canal antes do RRF", "",
                   "Textos completos, IDs, scores, notas e indicação de inclusão no contexto. "
                   "O RRF foi calculado sobre todos os chunks, sem limitar candidatos a estes top-100.", "",
                   "| Pergunta | Semântico | BM25 |", "|---|---|---|"]
    for question in questions:
        stem = f"{int(question['id']):02}"
        index_lines.append(f"| {question['id']} | [Markdown](semantic/{stem}.md) · [JSON](semantic/{stem}.json) | "
                           f"[Markdown](bm25/{stem}.md) · [JSON](bm25/{stem}.json) |")
    (args.output_dir / "top100" / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["context"], indent=2), flush=True)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
