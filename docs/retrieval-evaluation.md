# Avaliação local de recuperação

## Reranking opcional por cross-encoder

`src/reranking.py` implementa o reranking de um prefixo do ranking híbrido.
`ChunkReranker` usa `sentence_transformers.CrossEncoder`, modelo local, CPU,
comprimento máximo de 512 tokens e batches de 16 pares pergunta–chunk. Os
scores são logits sem ativação; eles ordenam apenas os candidatos pontuados.
Scores CE e RRF não são somados nem comparados. Empates CE preservam a ordem
híbrida. A cauda não pontuada é anexada na ordem original. O orçamento de
contexto e o texto dos chunks permanecem os mesmos.

Para a comparação congelada de 50, 100 e 200 candidatos:

```bash
.venv/bin/python tools/compare_reranking.py \
  --model models/ms-marco-MiniLM-L6-v2 \
  --candidates 50 100 200 --threads 4 \
  --output-dir experiments/runs/public22-ce-minilm-l6-2026-11-13
```

O diretório de modelo deve conter os pesos e tokenizer locais do
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
Nesta máquina foi usada uma cópia do snapshot já existente no cache, revisão
`233902d25c440f23af6f7d6e94d2946bac0bee0a`; hashes estão em
`models/ms-marco-MiniLM-L6-v2/provenance.json`. Não houve download de novos pesos.
O carregamento usa `local_files_only=True` e não requer novas dependências.

A execução preserva BM25 k1=1,25/b=0,75, RRF 60 e pesos 1/0,75, não usa KG e
reutiliza a extração/embeddings densos da execução original. Pares do maior
prefixo são pontuados uma vez por pergunta e os scores são reutilizados nos
menores prefixos. Portanto o tempo total da comparação não é a soma do custo
de quatro pipelines completos e não fornece a latência individual de 50/100.

Cada run contém ranking completo com `hybrid_rank` e `rerank_score`, contexto,
métricas, diagnóstico e manifesto. `score` continua sendo o score híbrido
original para rastreabilidade, mesmo depois da reordenação. `scores.json`
registra todos os scores CE calculados, tamanhos dos pares e tempos por pergunta.
Truncamento em 512 tokens ocorre somente para a pontuação; o contexto usa o
chunk original, sujeito à regra de orçamento já existente.

Além das métricas documentais, há nDCG@10 por chunk com **grau herdado do
documento** e contagem de chunks de documentos grau 2 nos primeiros dez. Isso é
um proxy, não julgamento de relevância de passagens: referências, métodos ou
introduções do artigo podem não sustentar a resposta. O diagnóstico informa
pares truncados, chunks da cauda não pontuada e composição do contexto por grau.

Para ativar o mesmo reranker em uma execução local completa:

```bash
.venv/bin/python tools/run_retrieval.py \
  --method hybrid --k1 1.25 --b 0.75 --rrf-k 60 \
  --dense-weight 1 --bm25-weight 0.75 \
  --reranker-model models/ms-marco-MiniLM-L6-v2 --rerank-candidates 100 \
  --tokenizer models/olmo-tokenizer \
  --output-dir experiments/runs/hybrid-ce-minilm-top100
```

O `main.py` de submissão permanece com o híbrido sem reranker. Para uma futura
submissão com CE será necessário habilitar a opção na inferência e incluir o
módulo e seus pesos no build; o ZIP atual não contém esses arquivos. Os pesos
locais adicionais somam cerca de 91,8 MB, mas o ZIP final dessa variante ainda
não foi construído. Inferência CPU evita VRAM adicional; os limites de tempo
total na H100 precisam de avaliação completa com o LLM.

## Escopo e dados

A amostra ativa contém 22 perguntas. Os IDs 16, 18 e 21 estão excluídos e os
demais IDs não são renumerados. O arquivo histórico atualmente localizado em
`experiments/judments/public-22-questions.json` ainda contém **25 perguntas** e
64 documentos: o nome não define seu conteúdo. Ele é preservado como fonte.

Gere o conjunto derivado com Python 3, a partir da raiz do repositório:

```bash
python3 tools/prepare_evaluation.py
```

`experiments/evaluation/public-22/` recebe `questions.json`, `judgments.json` e
`manifest.json`. O preparo verifica IDs, enunciados e SHA-256 de todos os PDFs.
Somente os documentos disponíveis entram nos denominadores das métricas. O
manifesto informa fontes ausentes e perguntas sem documentos de grau 2. Cada
execução de recuperação repete essa preparação para refletir o corpus atual.
PDFs em subdiretórios são encontrados recursivamente; nomes sem extensão devem
ser IDs únicos, como `AI-001` e `DONES-001`.

Os julgamentos são estimativas da IA, com negativos provisórios. Grau 2 indica
suporte a pelo menos parte da resposta. Os offsets de Markdown históricos não
são aplicados às novas extrações. Perguntas e respostas de referência não são
indexadas nem enviadas ao embedder como documentos.

## Executar o baseline sem o LLM final

Use o ambiente com as dependências oficiais de
`IEEE-CIS-DSQA-Starter-Kit/requirements-participant.txt`. Não há novas dependências.
Os modelos auxiliares devem estar disponíveis localmente: MiniLM em
`src/embed_model/` e artefatos de layout Docling em `src/docling_models/`.
Esses são os diretórios produzidos por `src/build.sh`; esse script baixa modelos
e recria esses diretórios, portanto é uma preparação com internet, fora da avaliação.
Também é possível fornecer diretórios existentes por `--embed-model` e
`--docling-models`, sem executar o build.

Para reproduzir a seleção de contexto, use o **tokenizer do OLMo oficial** em um
diretório local. Não é necessário ter os pesos de 32B:

```bash
.venv/bin/python tools/run_retrieval.py \
  --tokenizer /caminho/para/tokenizer-olmo \
  --output-dir experiments/runs/baseline-public22
```

Para avaliar somente o ranking, sem tokenizer:

```bash
.venv/bin/python tools/run_retrieval.py \
  --ranking-only \
  --output-dir experiments/runs/baseline-public22-ranking
```

## BM25 e RRF

Baixe somente os arquivos do tokenizer oficial, sem os pesos do LLM:

```bash
.venv/bin/python tools/download_tokenizer.py
```

O download usa `allenai/Olmo-3.1-32B-Instruct`, resolve uma revisão imutável e
registra revisão e hashes em `models/olmo-tokenizer/download.json`. A pasta de
modelos fica fora do controle de versão. O download requer internet apenas na
preparação; as execuções subsequentes usam o tokenizer local.

Para uma execução completa de parsing, embeddings e recuperação híbrida:

```bash
.venv/bin/python tools/run_retrieval.py \
  --method hybrid --k1 1.25 --b 0.75 --rrf-k 60 \
  --dense-weight 1 --bm25-weight 0.75 \
  --tokenizer models/olmo-tokenizer \
  --output-dir experiments/runs/hybrid-public22
```

`--method dense` permanece o padrão; `--method bm25` usa a ordenação lexical.
O construtor de índice completo ainda calcula embeddings mesmo no experimento
BM25, para manter o mesmo pipeline de extração. Não usar o tempo dessa variante
como estimativa de um sistema BM25 independente.

Para comparar os três métodos sem repetir parsing e embeddings, reutilize uma
execução densa completa (com `ranking.json`, `judgments.json` e `manifest.json`):

```bash
.venv/bin/python tools/compare_retrieval.py \
  --source-run experiments/runs/baseline-public22-ranking \
  --tokenizer models/olmo-tokenizer \
  --k1 1.25 --b 0.75 --rrf-k 60 \
  --output-dir experiments/runs/bm25-rrf-public22-initial
```

O replay valida os PDFs, julgamentos, enunciados, catálogo integral dos chunks e
a reprodução exata da ordem densa de cada pergunta. Ele reutiliza os scores
densos salvos, mas calcula os scores BM25 e a seleção de contexto com o
tokenizer real. Exporta subdiretórios `dense/`, `bm25/` e `hybrid/`, além de
`summary.json` e um manifesto de proveniência e recursos. Os tempos do replay
excluem extração e inferência de embeddings.

BM25 usa `rank-bm25==0.2.2`, já disponível no ambiente oficial, com Unicode
`\w+` e `casefold`, sem stemming nem remoção de stopwords. Termos da consulta
recebem a mesma tokenização. A fusão soma `1 / (60 + posição)` por canal, com
posições começando em 1 e pesos iguais por padrão. Os argumentos
`--dense-weight` e `--bm25-weight` permitem ponderar os canais; o score passa
a ser a soma de `peso_do_canal / (rrf_k + posição)`. Todos os chunks participam do canal
denso; apenas matches BM25 com score positivo recebem voto lexical. Isso evita
votos arbitrários dos empates em zero quando não há correspondência lexical.
Empates BM25/RRF seguem a ordem original do catálogo. Não há corte fixo de
candidatos nem limite de dez chunks no contexto.

O arquivo `context-metrics.json` da comparação inclui
`macro_full_context_document_coverage`, que considera **todos** os chunks
selecionados, além das métricas com cortes fixos. As variantes compartilham
exatamente a política de orçamento do baseline.

Na configuração de submissão escolhida em 10/11/2026, `src/main.py` usa
`RETRIEVAL_METHOD = "hybrid"`, `BM25_K1 = 1.25`, `BM25_B = 0.75`, `RRF_K = 60`,
`RRF_DENSE_WEIGHT = 1` e `RRF_BM25_WEIGHT = 0.75`. O avaliador local e as APIs de
recuperação mantêm o padrão denso; o starter kit original também foi preservado.
Nenhuma variante foi enviada automaticamente à plataforma da competição.

Para empacotar os modelos auxiliares já disponíveis, sem download:

```bash
bash src/build.sh --offline ../submissions/nome-da-submissao.zip
```

O caminho de saída é relativo a `src/`. O build recusa sobrescrever um ZIP
existente, confere integridade e tamanho máximo de 1 GB e inclui `main.py`,
`retrieval.py`, `embed_model/` e `docling_models/` na raiz do pacote.

## Grade de parâmetros com nomes explícitos

```bash
.venv/bin/python tools/sweep_retrieval.py \
  --source-run experiments/runs/baseline-public22-ranking \
  --tokenizer models/olmo-tokenizer \
  --output-dir experiments/runs/public22-bm25-rrf-sweep
```

Os padrões geram 91 runs: uma referência densa, nove controles BM25 e o produto
cartesiano de 81 configurações híbridas:

- `k1`: 0,75, 1,25 e 1,75.
- `b`: 0,5, 0,75 e 1.
- Peso BM25: 1, 0,75 e 0,5; peso denso fixo em 1.
- Constante RRF: 30, 60 e 90.

As listas são configuráveis por `--k1`, `--b`, `--bm25-weight` e `--rrf-k`, cada
uma aceitando múltiplos valores. A mesma execução valida o conjunto congelado
uma vez e calcula os tokens de cada chunk uma vez. Os scores BM25 são
reutilizados entre combinações com o mesmo `k1` e `b`. Esse cache não muda a
seleção de contexto, inclusive o fallback de truncamento.

Exemplos de nomes:

- `dense-reference`
- `bm25-k1_1p25-b0p75`
- `hybrid-dw1-bw0p75-k1_1p25-b0p75-rrf60`
- `hybrid-dw1-bw0p5-k1_0p75-b0p5-rrf30`

Cada subdiretório contém `manifest.json`, `summary.json`, ranking, contexto e
métricas. O ranking completo usa `chunk_ids` e uma lista paralela de `scores`;
o texto e os tokens de todos os chunks ficam no catálogo compartilhado
`chunks.jsonl` da grade. O contexto renderizado é salvo integralmente em cada
run. Os manifestos referenciam hashes do catálogo, dados, tokenizer e código.

`summary.json` e `summary.csv` no diretório da grade consolidam as configurações,
nDCG@10, recall no contexto completo, recall por domínio, contagem de chunks,
tokens e tempo incremental. As métricas têm o mesmo significado e limites das
comparações anteriores. Tempos com caches não incluem parsing/embeddings nem
podem ser comparados como latências completas de submissão.

Os resultados dessa grade usam as mesmas 22 perguntas de desenvolvimento. Eles
servem para selecionar candidatas a novos testes; a melhor configuração local
não é uma estimativa independente de desempenho no benchmark oculto.

Escolha uma pasta de saída nova ou vazia para cada execução. As duas opções são
mutuamente exclusivas. O tokenizer e o embedder são carregados apenas de arquivos
locais. O pipeline de recuperação não carrega o modelo gerador.

## Saídas e interpretação

- `ranking.json`: todos os chunks ordenados por similaridade, com IDs de chunk
  e documento, caminho relativo, texto, score e tempo por pergunta.
- `ranking-metrics.json`: nDCG, recall, precisão e MRR documental em 1, 3, 5 e 10;
  também recall de documentos dentro dos primeiros k chunks.
- `context.json` e `context-metrics.json`: disponíveis com `--tokenizer`;
  contêm o contexto selecionado e as mesmas métricas restritas a essa seleção.
- `questions.json`, `judgments.json` e `manifest.json`: conjunto efetivamente
  usado, hashes dos dados de origem, cobertura, falhas de parsing, configuração,
  duração, bytes dos embeddings e pico de RSS do processo Linux. RSS exclui
  processos filhos e memória da GPU; não é uma medição completa de recursos.

No ranking documental, repetições do documento são removidas pela primeira
ocorrência. Na cobertura dos primeiros k chunks, as repetições ocupam posições.
`ge1` aceita graus 1 e 2; `ge2` aceita somente grau 2. Recall e MRR sem documentos
relevantes são `null`, e a média informa quantas perguntas têm denominador
definido. Precisão divide por k mesmo quando há menos resultados.

A seleção de contexto conserva a regra original: para no primeiro chunk que
excede o orçamento. Se o primeiro chunk exceder sozinho, usa truncamento
aproximado por caracteres. O orçamento de 10 mil tokens não inclui cabeçalhos,
pergunta ou template de chat; isso não constitui um limite estrito do prompt.

As métricas do contexto continuam sendo **documentais**, mesmo quando um chunk
foi truncado. Não medem cobertura de fatos, suficiência da passagem ou acurácia
das respostas. Nenhum resultado de recuperação deve ser apresentado como
acurácia oficial.

Para recalcular métricas de uma execução existente:

```bash
python3 tools/evaluate_retrieval.py \
  --input experiments/runs/baseline-public22/context.json \
  --judgments experiments/runs/baseline-public22/judgments.json \
  --output experiments/runs/baseline-public22/context-metrics.json
```

O avaliador rejeita perguntas duplicadas, documentos não julgados e conjuntos
incompletos por padrão. `--allow-partial` permite explicitamente uma análise
parcial e registra essa escolha. Não há julgamentos por chunk implementados.

## Verificação e submissão

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Os testes conferem equivalência de chunking e contexto com o starter original,
métricas com rankings sintéticos, exclusão de perguntas, fontes ausentes,
hashes, descoberta recursiva, fórmula BM25, votos RRF e ausência de corte em
dez chunks. Rankings sintéticos não são benchmarks.

A interface oficial `answer(question, corpus_dir, model_id) -> str` permanece.
`src/build.sh` inclui `retrieval.py` no ZIP. Os dados e relatórios locais não são
incluídos na submissão. Os pesos auxiliares e a configuração densa permanecem
os mesmos; o módulo adicional é pequeno. Nenhum ganho de tempo ou acurácia foi
medido, nem foi validado o limite total de 60 minutos na H100. A descoberta
recursiva processa todos os PDFs locais, incluindo subdiretórios antes ignorados.
