# Grade BM25 e RRF ponderado — 22 perguntas

Executada em 10/11/2026: **91 runs completas**, com 22 perguntas cada, nos mesmos
5.174 chunks dos 64 PDFs. O tokenizer e os scores densos são os da comparação
inicial. Nenhum LLM gerador foi carregado. A referência densa e os controles
BM25/híbrido padrão reproduziram exatamente os contextos da comparação anterior.

## Grade e artefatos

- Uma referência densa e nove controles BM25.
- 81 híbridos: `k1 ∈ {0.75, 1.25, 1.75}`, `b ∈ {0.5, 0.75, 1}`,
  peso BM25 em `{1, 0.75, 0.5}`, constante RRF em `{30, 60, 90}`.
- Peso denso fixo em 1. O score soma `peso / (rrf_k + posição)` por canal,
  começando em 1. Apenas scores BM25 positivos recebem voto lexical.
- Tokenização lexical, chunking e regra de seleção de contexto preservados.
  Orçamento de 10.000 tokens de texto dos chunks, sem corte fixo em dez chunks.

Os resultados consolidados, incluídos no controle de versão, estão em
[summary.csv](../../experiments/results/public22-bm25-rrf-sweep/summary.csv),
[summary.json](../../experiments/results/public22-bm25-rrf-sweep/summary.json) e
[manifest.json](../../experiments/results/public22-bm25-rrf-sweep/manifest.json).

As runs completas ficam em `experiments/runs/public22-bm25-rrf-sweep/`, fora do
controle de versão. Cada run tem nome explícito, manifesto, ranking completo,
contexto renderizado e métricas. Exemplos:

- `dense-reference`
- `bm25-k1_1p25-b0p75`
- `hybrid-dw1-bw0p75-k1_1p25-b0p75-rrf60`
- `hybrid-dw1-bw0p5-k1_1p75-b0p75-rrf30`

O catálogo compartilhado `chunks.jsonl` contém os textos originais e os tokens.
Cada ranking referencia os IDs desse catálogo, com uma lista paralela de scores.
O manifesto registra hashes dos dados, catálogo, tokenizer e código. Os tempos
e recursos abaixo correspondem à execução completa bem-sucedida.

```bash
.venv/bin/python tools/sweep_retrieval.py \
  --source-run experiments/runs/baseline-public22-ranking \
  --tokenizer models/olmo-tokenizer \
  --output-dir experiments/runs/public22-bm25-rrf-sweep
```

Escolher outro diretório para repetir a grade existente. As listas da grade
podem ser substituídas com `--k1`, `--b`, `--bm25-weight` e `--rrf-k`.

## Efeito isolado do peso BM25

Mantendo `k1=1.25`, `b=0.75` e RRF 60:

| Peso BM25 | nDCG documental@10 | Recall grau 2 no contexto completo | Recall graus 1–2 no contexto completo |
| ---: | ---: | ---: | ---: |
| 1 | 0,8741 | 0,9773 | 0,9784 |
| 0,75 | 0,8816 | 0,9773 | 0,9784 |
| 0,5 | 0,8779 | 0,9773 | 0,9784 |

Reduzir o peso para 0,75 melhorou o nDCG nessa comparação controlada. Os
resultados da grade completa mostram interações com `k1`, `b` e a constante
RRF; não há um peso isoladamente superior em todas as combinações.

## Melhores nDCGs da grade

Todos os pesos densos são 1. Recall considera documentos distintos presentes
em **todo** o contexto selecionado, e não somente os primeiros dez chunks.

| Peso BM25 | k1 | b | RRF | nDCG documental@10 | Recall grau 2 | Recall graus 1–2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5 | 1,75 | 0,75 | 30 | **0,9085** | 0,9773 | 0,9556 |
| 0,75 | 0,75 | 1 | 90 | 0,8997 | 0,9773 | 0,9784 |
| 0,75 | 1,25 | 1 | 90 | 0,8996 | 0,9773 | 0,9784 |
| 0,75 | 1,75 | 1 | 90 | 0,8981 | 0,9773 | 0,9708 |
| 1 | 1,25 | 1 | 90 | 0,8981 | 0,9773 | 0,9784 |

O melhor nDCG foi da run
`hybrid-dw1-bw0p5-k1_1p75-b0p75-rrf30`: aumento absoluto de 0,0345 sobre o
híbrido inicial (0,8741) e de 0,0627 sobre o denso (0,8458). Ela selecionou em
média 54,95 chunks e 9.890,59 tokens de texto por pergunta.

Em relação ao híbrido inicial, essa run melhorou nDCG em seis perguntas,
piorou em três e empatou em treze. As mudanças de nDCG ficaram nas perguntas
DONES. O ganho de ordenação veio acompanhado de queda de 0,9784 para 0,9556 no
recall que também inclui documentos de grau 1.

Por isso, vale manter também a candidata
`hybrid-dw1-bw0p75-k1_0p75-b1-rrf90`: nDCG 0,8997, com a cobertura mais ampla
preservada em 0,9784. A grade não estabelece qual delas gerará melhores respostas.

Todos os **81 híbridos** tiveram recall de grau 2 igual a 0,9773 no contexto
completo: 1,0000 em DONES e 0,9583 em IA. Nenhuma dessas combinações resolveu
a ausência de `AI-006` na pergunta 12. Todas as 91 runs respeitaram o orçamento
de tokens de texto dos chunks; cabeçalhos, pergunta e template de chat continuam
fora desse orçamento, como no baseline.

## Recursos, correção e limites

A grade bem-sucedida levou **80,51 s**, com pico de RSS de aproximadamente
**1,09 GiB** no processo Linux. Textos, contagens de tokens e scores densos
foram reutilizados; scores BM25 foram compartilhados entre configurações com
os mesmos parâmetros. Esses tempos excluem parsing, inferência de embeddings,
carregamento e geração do LLM, e não validam os 60 minutos de avaliação na H100.

Uma tentativa anterior foi interrompida em `b=1`: dois chunks contêm apenas
símbolos e não geram tokens lexicais. A biblioteca calculava `0/0` para esses
chunks. O adaptador agora retorna zero para documentos lexicalmente vazios,
mantendo-os nas estatísticas do corpus para preservar os scores anteriores.
Um teste de regressão cobre esse caso. A tentativa incompleta foi preservada
em `public22-bm25-rrf-sweep-interrupted-b1-empty-chunks/`, com estado `failed`;
seus resultados não entram nos consolidados.

Passaram **25 testes**, incluindo fórmula RRF ponderada, caches de tokens,
caso `b=1`, nomes únicos e geração de artefatos da grade. Foram verificados os
manifestos e as 22 perguntas de cada uma das 91 runs concluídas.

Nenhuma dependência ou peso adicional foi introduzido. A grade é uma ferramenta
local de experimentação e seus artefatos não entram no ZIP. A avaliação oficial
executará apenas a configuração escolhida, não a grade. O baseline de inferência
continua denso; o tamanho final do ZIP e o tempo total com o LLM não foram medidos.

Os julgamentos são estimativas de IA no nível de documento. Grau 2 pode apoiar
apenas parte da resposta e não garante que a passagem necessária esteja no
contexto. A escolha entre 81 combinações nas mesmas 22 perguntas é exploratória,
sem conjunto independente de validação; os ganhos locais não demonstram melhora
na acurácia oficial nem significância estatística.
