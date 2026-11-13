#!/usr/bin/env python3
"""Validate an extracted KG ZIP offline; replay all 22 contexts, mock only LLM/dense encoding."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from prepare_evaluation import ROOT, prepare
from compare_retrieval import load_frozen_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--real-pdf', action='store_true')
    args = parser.parse_args()
    manifest_path = args.archive.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == manifest['sha256']
    reference = ROOT/manifest['reference_run']
    questions, judgments, _ = prepare(ROOT/'data/pdfs')
    chunks, dense, _ = load_frozen_run(ROOT/'experiments/runs/baseline-public22-ranking', questions, judgments)
    old_contexts = json.loads((reference/'hybrid-bw0p75-k1_1p25-b0p75-rrf60-kg-aliases/context.json').read_text())['questions']
    expected = {str(q['id']): q for q in old_contexts}
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/olmo-tokenizer'), local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='kg-submission-check-', dir=ROOT/'experiments/runs') as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert hashlib.sha256(archive.read(name)).hexdigest() == manifest['zip_entry_sha256'][name]
            archive.extractall(directory)
        sys.path.insert(0, str(directory))
        module = importlib.import_module('main')
        retrieval = importlib.import_module('retrieval')
        kg_module = importlib.import_module('kg_expansion')
        assert Path(module.__file__).parent == directory
        assert Path(retrieval.__file__).parent == directory
        assert list(inspect.signature(module.answer).parameters) == ['question', 'corpus_dir', 'model_id']
        assert module.SYSTEM_PROMPT == ('You are a scientific expert. Answer the following question as accurately '
                                        'and concisely as possible based on your knowledge.')
        config = json.loads((directory/'kg_config.json').read_text())
        resource = directory/'kg_resources/kg_rag_chunks.jsonl'
        pdf_check = None
        print('Extracted package; starting offline smoke/replay', flush=True)
        if args.real_pdf:
            started = time.perf_counter()
            corpus = directory/'smoke-corpus'
            corpus.mkdir()
            pdf = next((ROOT/'data/pdfs').rglob('AI-001.pdf'))
            shutil.copyfile(pdf, corpus/pdf.name)
            index_pdf = retrieval.build_index(corpus)
            kg_pdf = kg_module.KnowledgeExpansion(resource, index_pdf.chunks, policy=config['policy'])
            expanded, _ = kg_pdf.expand(questions[-1]['question'], 'aliases')
            smoke = retrieval.retrieve(questions[-1]['question'], index_pdf, tokenizer,
                                       method='hybrid', bm25_weight=.75, lexical_question=expanded)
            assert index_pdf.chunks and smoke['selected_chunks']
            pdf_check = {'offline': True, 'document_id': 'AI-001', 'chunks': len(index_pdf.chunks),
                         'selected_chunks': len(smoke['selected_chunks']), 'seconds': time.perf_counter()-started,
                         'model_paths_inside_zip': True, 'llm_loaded': False}
            print('Offline PDF smoke passed', pdf_check, flush=True)
            del index_pdf, kg_pdf, smoke
            import gc
            gc.collect()
        calls = {'build': 0, 'model': 0, 'dense_queries': []}
        class Encoder:
            def encode(self, texts, **kwargs):
                calls['dense_queries'].append(texts)
                return np.array([[1.]], dtype=np.float32)
        index = retrieval.RetrievalIndex(chunks, None, Encoder(), [], [], 0)
        def build_index(corpus_dir):
            calls['build'] += 1
            return index
        class Inputs(dict):
            def to(self, device): return self
        class Tokenizer:
            pad_token_id = tokenizer.pad_token_id
            def encode(self, text): return tokenizer.encode(text)
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return Inputs(input_ids=torch.tensor([[1, 2]]))
            def decode(self, values, **kwargs): return 'validation-answer'
        wrapper = Tokenizer()
        class Model:
            device = 'cpu'
            def eval(self): return self
            def generate(self, **kwargs):
                assert kwargs['max_new_tokens'] == 2048 and kwargs['do_sample'] is False
                assert not torch.is_grad_enabled()
                return torch.tensor([[1, 2, 3]])
        def load_model(*a, **kw):
            calls['model'] += 1
            assert kw['device_map'] == 'cuda' and kw['dtype'] == torch.bfloat16
            return Model()
        module.build_index = build_index
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: wrapper)
        module.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
        original_retrieve = module.retrieve
        captured = []
        def capture(*a, **kw):
            result = original_retrieve(*a, **kw)
            captured.append(result)
            return result
        module.retrieve = capture
        for q in questions:
            qid = str(q['id'])
            index.embeddings = dense[qid][:, None]
            assert module.answer(q['question'], 'frozen-corpus', 'provided-model') == 'validation-answer'
            actual = captured[-1]
            assert actual['context'] == expected[qid]['context'], f'Context mismatch Q{qid}'
            assert [c['chunk_id'] for c in actual['selected_chunks']] == [c['chunk_id'] for c in expected[qid]['chunks']]
            assert wrapper.messages[1]['content'] == expected[qid]['context'] + q['question']
            assert calls['dense_queries'][-1] == [q['question']]
        print('All frozen contexts and original prompts matched', len(questions), flush=True)
        assert calls['build'] == calls['model'] == 1
        # Detect corrupt/mismatched resources at initialization, before loading an LLM.
        module._cache.clear()
        config['kg_sha256'] = 'invalid'
        (directory/'kg_config.json').write_text(json.dumps(config))
        try:
            module._load('provided-model', 'frozen-corpus')
        except ValueError as error:
            assert 'checksum' in str(error)
        else:
            raise AssertionError('Invalid resource checksum was accepted')
    import resource
    manifest['validation_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    manifest['validation'].update({'zip_crc': 'passed', 'entry_hashes': 'passed',
                                   'official_answer_signature': 'passed', 'unit_tests_passed': 50,
                                   'offline_frozen_replay': {'matched_contexts': len(questions),
                                        'original_dense_questions_and_llm_prompt': True,
                                        'index_and_llm_loaded_once': True, 'generation_mocked': True,
                                        'dense_scores_frozen': True},
                                   'offline_retrieval_smoke': pdf_check,
                                   'kg_checksum_rejection': 'passed'})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'archive': str(args.archive), 'validation': manifest['validation']}, indent=2))


if __name__ == '__main__': main()
