"""Optional local cross-encoder reranking; no generation or network access."""
from pathlib import Path
import logging
import numpy as np


class QueryTooLongError(ValueError):
    """The complete query leaves no room for a passage in the CE window."""


def reorder_prefix(ranking, scores):
    """Reorder scored prefix, append untouched tail, never compare CE/RRF scores."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) > len(ranking) or not np.isfinite(scores).all():
        raise ValueError('Invalid reranker score count or non-finite scores')
    order = np.argsort(-scores, kind='stable')
    prefix = [{**ranking[int(i)], 'hybrid_rank': ranking[int(i)]['rank'],
               'rerank_score': float(scores[i])} for i in order]
    tail = [{**row, 'hybrid_rank': row['rank']} for row in ranking[len(scores):]]
    return [{**row, 'rank': rank} for rank, row in enumerate(prefix + tail, 1)]


class ChunkReranker:
    def __init__(self, model_path, device='cpu', max_length=512, batch_size=16):
        if not Path(model_path).is_dir():
            raise ValueError(f'Local reranker model directory missing: {model_path}')
        if max_length <= 0 or batch_size <= 0:
            raise ValueError('Length and batch size must be positive')
        import torch
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_path), device=device, max_length=max_length,
                                  local_files_only=True, activation_fn=torch.nn.Identity())
        self.max_length, self.batch_size = max_length, batch_size

    def query_diagnostics(self, question):
        """Reserve the complete query and pair markers before passage tokens."""
        tokenizer = self.model.tokenizer
        query_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        passage_budget = (self.max_length - query_tokens
                          - tokenizer.num_special_tokens_to_add(pair=True))
        return {'query_tokens': query_tokens, 'passage_token_budget': passage_budget,
                'fallback_to_hybrid': passage_budget <= 0}

    def score(self, question, chunks):
        if not chunks:
            return np.empty(0)
        diagnostics = self.query_diagnostics(question)
        if diagnostics['fallback_to_hybrid']:
            raise QueryTooLongError('Complete query leaves no passage tokens in the CE window')
        return np.asarray(self.model.predict([(question, c['text']) for c in chunks],
                          batch_size=self.batch_size, show_progress_bar=False,
                          convert_to_numpy=True,
                          processing_kwargs={'text': {'truncation': 'only_second',
                                                      'max_length': self.max_length}})).reshape(-1)

    def rerank(self, question, ranking, candidates=100):
        if candidates <= 0:
            raise ValueError('Candidate count must be positive')
        prefix = ranking[:candidates]
        try:
            scores = self.score(question, prefix)
        except QueryTooLongError:
            logging.getLogger(__name__).warning(
                'Skipping CE: complete query leaves no passage tokens; preserving input ranking')
            return [dict(row) for row in ranking]
        if len(scores) != len(prefix):
            raise ValueError('Cross-encoder returned an incorrect number of scores')
        return reorder_prefix(ranking, scores)
