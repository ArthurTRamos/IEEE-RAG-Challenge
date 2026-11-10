# Avaliação local de recuperação

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
hashes e descoberta recursiva. Rankings sintéticos não são benchmarks.

A interface oficial `answer(question, corpus_dir, model_id) -> str` permanece.
`src/build.sh` inclui `retrieval.py` no ZIP. Os dados e relatórios locais não são
incluídos na submissão. Os pesos auxiliares e a configuração densa permanecem
os mesmos; o módulo adicional é pequeno. Nenhum ganho de tempo ou acurácia foi
medido, nem foi validado o limite total de 60 minutos na H100. A descoberta
recursiva processa todos os PDFs locais, incluindo subdiretórios antes ignorados.
