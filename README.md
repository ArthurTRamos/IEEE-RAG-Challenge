# IEEE-RAG-Challenge

QA científico para o IEEE CIS GPAIS Technical Challenge. Leia as instruções em
[AGENTS.md](AGENTS.md) e os [requisitos](docs/competition-requirements.md).

O starter kit original fica em `IEEE-CIS-DSQA-Starter-Kit/`. A implementação em
`src/` compartilha a recuperação entre a submissão e a avaliação local, mantendo
o baseline denso: MiniLM na CPU, Docling sem OCR/estrutura de tabelas, chunks de
800 caracteres com sobreposição de 150 e contexto com orçamento de 10 mil tokens
de texto dos chunks. O avaliador local mantém o padrão denso. A configuração
atual de submissão em `src/main.py` usa RRF híbrido com BM25 `k1=1.25`, `b=0.75`,
constante RRF 60 e pesos denso/BM25 de 1/0.75, sem reranking ou grafos.

Veja [como executar a avaliação de recuperação](docs/retrieval-evaluation.md)
para as 22 perguntas ativas, sem carregar os pesos do LLM final.
