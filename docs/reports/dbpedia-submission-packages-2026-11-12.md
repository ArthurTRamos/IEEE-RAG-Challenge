# Pacotes de submissão DBpedia — 12/11/2026

Os dois ZIPs foram gerados e validados após a retomada do WSL. Os arquivos
sobreviveram à queda com SHA-256 igual ao registrado na construção e CRC válido.
A validação interrompida foi refeita sequencialmente, extraindo os modelos para
o disco do projeto e preservando logs em
`experiments/runs/kg-package-validation-2026-11-12/`.

| Variante | Bytes | SHA-256 |
| --- | ---: | --- |
| [original-legacy](../../submissions/2026-11-12-dbpedia-original-legacy.zip) | 399,603,332 | `a8bf22b59aec7d71609736c34abbf6020a108900960588d6371472a81d09df42` |
| [v2-contextual](../../submissions/2026-11-12-dbpedia-v2-contextual.zip) | 400,018,521 | `4b933399eda72513053ff4497f82ee833972a90ac36c051d1be69257dcfe9b44` |

## Configurações congeladas

- **original-legacy:** `data/kg_rag_chunks.jsonl`, política `legacy`, modo
  `aliases`. Referência: `public22-kg-legacy-replay-2026-11-12`, nDCG 0,8812.
- **v2-contextual:** `experiments/runs/dbpedia-reviewed-v2/kg_rag_chunks.jsonl`,
  política `contextual`, modo `aliases`. Referência:
  `public22-v2-contextual-decision-2026-11-12`, nDCG 0,8671.

Ambas preservam o MiniLM, Docling e seus pesos auxiliares com hashes idênticos
à submissão de score informado 60. Preservam prompt, chunking, geração greedy,
máximo de 2.048 novos tokens, contexto de 10.000 tokens, BM25 k1=1,25/b=0,75,
RRF 60 e pesos denso/BM25 1/0,75. A expansão afeta apenas a consulta BM25;
encoder e pergunta do prompt usam a pergunta original. Não há expansão de arestas.

`src/main_kg.py` é o template empacotado como `main.py`, com a assinatura oficial
`answer(question, corpus_dir, model_id)`. Cada pacote inclui `retrieval.py`,
`kg_expansion.py`, `kg_config.json`, o JSONL em `kg_resources/` e os mesmos modelos
auxiliares do baseline. A configuração inclui fonte DBpedia e checksum do recurso.
O índice de entidades é construído uma vez sobre o corpus recebido. O OLMo
continua sendo o modelo fornecido pelo ambiente; não faz parte do ZIP.

Não entram notebooks, PDFs de teste, perguntas, julgamentos, cache SQLite,
Turtle, scripts de construção ou dependências novas. O baseline `src/main.py`
não foi alterado. O parâmetro opcional `lexical_question` em `retrieve()` mantém
o comportamento anterior quando ausente.

## Validação

- 50 testes unitários passaram, incluindo separação entre consulta densa e lexical.
- CRC, SHA-256 global e hashes de todas as 26 entradas passaram em cada ZIP.
- Foram verificados assinatura oficial, prompt e parâmetros de geração.
- Executando os módulos extraídos de cada ZIP, os 22 contextos e IDs dos chunks
  selecionados coincidiram exatamente com as respectivas referências locais.
- No replay, somente scores densos e geração foram simulados. A geração simulada
  confirmou a preservação da pergunta no prompt e os parâmetros de inferência;
  o índice e o carregador simulado do modelo foram chamados uma vez por corpus.
- Cada pacote também processou `AI-001.pdf` com seus próprios modelos locais,
  offline, indexando 112 chunks e selecionando 33 para a pergunta de smoke test.
  Esse teste levou aproximadamente 40,32 s / 39,23 s, respectivamente; não é
  uma medição das 100 perguntas ou do desempenho final na H100.
- Um checksum incorreto de KG é rejeitado na inicialização.

Os resultados de validação estão nos manifestos ao lado dos ZIPs e em
`experiments/submissions/2026-11-12-dbpedia-*.json`.

## Limites e envio

Cada arquivo tem aproximadamente 400 MB, abaixo de 1 GB. Não foi executada
inferência real do OLMo nem medida a execução total na H100; o limite de 60
minutos ainda precisa ser confirmado no ambiente oficial. Na etapa de construção, não houve envio à plataforma; posteriormente o
usuário informou a avaliação da variante legacy, registrada abaixo. Os scores locais são métricas de recuperação documental, não
acurácia de respostas nem previsão de score oficial.

## Construção e validação reproduzíveis

```bash
python3 tools/build_kg_submissions.py --output-dir submissions
```

O construtor recusa sobrescrever os pacotes existentes; escolher outro diretório
para uma nova construção. Ele reutiliza os modelos locais e confere os hashes
com o manifesto do baseline. Não faz downloads.

Para validar, executar uma variante por vez. Exemplo:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/validate_kg_submission.py \
  --archive submissions/2026-11-12-dbpedia-original-legacy.zip --real-pdf
```

Usar o outro nome de ZIP para validar a contextual. O validador força o modo
offline e usa armazenamento temporário em `experiments/runs/`, evitando extrair
pesos para o tmpfs. Nenhum teste substitui a execução completa na H100.

## Resultado posterior informado pelo usuário

A variante `original-legacy` foi submetida e obteve **57%**, contra **60%** do
híbrido: −3 pontos percentuais. O ZIP permanece inalterado. Os manifestos foram
atualizados para `evaluated_user_report`, com resultado sem verificação
independente do painel. O tempo de execução não foi fornecido. Em 13/11/2026, o usuário informou **60%** para `v2-contextual`, empate com
o híbrido e +3 pontos percentuais frente à legacy. Tempo e resultados por
pergunta não foram fornecidos. O manifesto foi atualizado para
`evaluated_user_report`; os ZIPs permanecem inalterados.
Ver [histórico de submissões](submission-results.md).
