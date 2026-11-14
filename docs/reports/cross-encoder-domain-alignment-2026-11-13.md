# Alinhamento de domínio: híbrido e cross-encoder

Análise das 22 perguntas, usando os rankings congelados da execução de 13/11/2026. Sem nova inferência, alteração de ranking ou filtragem de domínio.

Os domínios vêm dos metadados dos documentos em `public-22-questions.json`, conferidos pelos hashes dos PDFs. `AI Papers` é agrupado como IA/matemática; `DONES` como IFMIF-DONES. A pergunta usa o domínio da amostra pública. Trata-se de domínio do documento de origem, não de classificação semântica do conteúdo de cada passagem.

## Resultados agregados

As frações abaixo são médias das frações por pergunta (macro). Micro também é exportado. No top-50, todos os denominadores são 50; no contexto efetivo, variam conforme o orçamento de tokens.

| Variante | Mesmo domínio no top-50 | Fração top-50 | Mesmo domínio no contexto efetivo | Fração no contexto efetivo |
| --- | ---: | ---: | ---: | ---: |
| Híbrido | 49.82/50 | 99.64% | 52.41/52.82 | 99.33% |
| CE sobre 50 | 49.82/50 | 99.64% | 51.86/52.27 | 99.33% |
| CE sobre 100 | 49.86/50 | 99.73% | 49.41/49.59 | 99.66% |
| CE sobre 200 | 49.82/50 | 99.64% | 48.95/49.18 | 99.57% |

## Top-50 por pergunta

Cada célula conta chunks do domínio da pergunta entre os primeiros 50. Não representa chunks relevantes ou fatos sustentados.

| Pergunta | Domínio | Híbrido | CE 50 | CE 100 | CE 200 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | DONES | 50 | 50 | 50 | 50 |
| 2 | DONES | 50 | 50 | 50 | 50 |
| 3 | DONES | 48 | 48 | 48 | 47 |
| 4 | DONES | 50 | 50 | 50 | 50 |
| 5 | DONES | 50 | 50 | 50 | 50 |
| 6 | DONES | 50 | 50 | 50 | 50 |
| 7 | DONES | 50 | 50 | 50 | 50 |
| 8 | DONES | 50 | 50 | 50 | 50 |
| 9 | DONES | 50 | 50 | 50 | 50 |
| 10 | DONES | 50 | 50 | 50 | 50 |
| 11 | IA/matemática | 50 | 50 | 50 | 50 |
| 12 | IA/matemática | 50 | 50 | 49 | 49 |
| 13 | IA/matemática | 50 | 50 | 50 | 50 |
| 14 | IA/matemática | 50 | 50 | 50 | 50 |
| 15 | IA/matemática | 50 | 50 | 50 | 50 |
| 17 | IA/matemática | 50 | 50 | 50 | 50 |
| 19 | IA/matemática | 48 | 48 | 50 | 50 |
| 20 | IA/matemática | 50 | 50 | 50 | 50 |
| 22 | IA/matemática | 50 | 50 | 50 | 50 |
| 23 | IA/matemática | 50 | 50 | 50 | 50 |
| 24 | IA/matemática | 50 | 50 | 50 | 50 |
| 25 | IA/matemática | 50 | 50 | 50 | 50 |

## Leitura dos resultados

- O híbrido já tem 1096/1100 ocorrências de chunks no domínio correto no top-50: somente quatro vêm do outro domínio, distribuídas entre as perguntas 3 e 19.
- CE sobre 50 não muda o conjunto do top-50; por isso as contagens são idênticas.
- CE sobre 100 elimina as duas ocorrências de outro domínio da pergunta 19 e introduz uma na pergunta 12. A pergunta 3 permanece com duas. A fração agregada aumenta somente 0,09 ponto percentual.
- CE sobre 200 mantém a melhora na pergunta 19, mas introduz uma ocorrência na pergunta 12 e mais uma na pergunta 3; o total agregado volta ao do híbrido.
- No contexto efetivo, CE 100 melhora a fração de domínio em três perguntas, piora em duas e empata em dezessete. As contagens absolutas dependem também dos comprimentos dos chunks selecionados.
- O híbrido tem, em média, 16,41 chunks do domínio correto, mas de documentos grau 0, no top-50. No CE 100 são 15,36. Isso mostra que o problema remanescente é predominantemente seleção dentro do domínio, segundo os julgamentos atuais.
- Nesta amostra, não houve ocorrências de chunks do outro domínio provenientes de documentos com grau 1 ou 2 nos conjuntos analisados. Isso não garante que evidência entre domínios seja sempre inútil em outras perguntas.

A métrica está próxima da saturação e não distingue, sozinha, as melhores configurações. O resultado não oferece motivação forte para um filtro rígido de domínio: o espaço de ganho observado é pequeno, e erros futuros de classificação poderiam retirar evidências úteis. Uma próxima hipótese mais alinhada a estes dados é melhorar a discriminação entre passagens do mesmo domínio, usando negativos difíceis e julgamentos de suporte factual por chunk. Isso ainda precisa de experimento; não foi implementado filtro nem treinamento.

## Artefatos e reprodução

```bash
.venv/bin/python tools/evaluate_domains.py \
  --k 50 \
  --output-dir experiments/results/public22-ce-domain-alignment-2026-11-13
```

Escolher uma pasta nova para repetir a execução. O [CSV por pergunta](../../experiments/results/public22-ce-domain-alignment-2026-11-13/per-question.csv) contém as 176 linhas (22 perguntas × quatro variantes × dois recortes), com contagens do mesmo/outro domínio, frações, cruzamento com relevância documental e diferenças em relação ao controle. [summary.json](../../experiments/results/public22-ce-domain-alignment-2026-11-13/summary.json) inclui agregados gerais e separados pelos dois domínios. O manifesto registra hashes dos dados, metadados e código.

Os julgamentos de relevância são estimativas de IA no nível de documento. Não se deve interpretar alinhamento de domínio como relevância de passagem nem converter estas frações em acurácia de QA. Não houve impacto na submissão de 1 GB ou no tempo de avaliação de 60 minutos; esta é uma análise offline dos artefatos existentes.
