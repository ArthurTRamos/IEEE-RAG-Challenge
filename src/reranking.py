"""Optional local cross-encoder reranking; no generation or network access."""
from pathlib import Path
import numpy as np


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

    def score(self, question, chunks):
        if not chunks:
            return np.empty(0)
        return np.asarray(self.model.predict([(question, c['text']) for c in chunks],
                          batch_size=self.batch_size, show_progress_bar=False,
                          convert_to_numpy=True)).reshape(-1)

    def rerank(self, question, ranking, candidates=100):
        if candidates <= 0:
            raise ValueError('Candidate count must be positive')
        prefix = ranking[:candidates]
        scores = self.score(question, prefix)
        if len(scores) != len(prefix):
            raise ValueError('Cross-encoder returned an incorrect number of scores')
        return reorder_prefix(ranking, scores)
