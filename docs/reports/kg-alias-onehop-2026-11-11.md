# KG: aliases, uma aresta e diagnóstico de seções

Comparação nas 22 perguntas ativas, sem geração pelo LLM. Foram reutilizados
os 5.174 chunks e os scores densos do baseline congelado. Nenhum texto, ID,
encoder, prompt, regra de orçamento ou parâmetro de fusão foi alterado.
O controle usa BM25 k1=1,25/b=0,75 e RRF 60 com pesos denso/BM25 de 1/0,75.

## Implementação e reprodução

```bash
.venv/bin/python tools/compare_kg.py \
  --output-dir experiments/runs/public22-kg-alias-onehop-2026-11-11-final
```

As três runs são `hybrid-bw0p75-k1_1p25-b0p75-rrf60-kg-control`,
`hybrid-bw0p75-k1_1p25-b0p75-rrf60-kg-aliases` e
`hybrid-bw0p75-k1_1p25-b0p75-rrf60-kg-one_hop`. Cada uma exporta manifesto,
ranking, contexto, métricas documentais, diagnóstico por pergunta e resumo.
Os consolidados estão em
[experiments/results/public22-kg-alias-onehop-2026-11-11](../../experiments/results/public22-kg-alias-onehop-2026-11-11/summary.json).

`src/kg_expansion.py` usa somente JSONL e bibliotecas disponíveis no ambiente.
A ligação é lexical por sequências de tokens completas, com labels e aliases;
aliases ambíguos são descartados. Termos de uma palavra com menos de quatro
caracteres são ignorados para reduzir associações espúrias. Há uma ligação
entidade–chunks para exigir suporte de menção no corpus, não confirmação factual.

Os limites fixos são oito entidades-semente, label e até dois aliases por
semente, oito vizinhos e oito palavras por expressão adicionada. Sementes são
selecionadas em ordem de URI, sem calibrar confiança ou usar julgamentos de
relevância. Vizinhos de saída são priorizados por número de sementes que os
alcançam, com penalização por grau e desempate por URI. Apenas entidades com
menções nos chunks participam. Não há expansão recursiva.

As consultas expandidas afetam **somente BM25**. A consulta densa e a pergunta
enviada à LLM permanecem originais; não há texto DBpedia no prompt. Aliases podem
repetir termos já existentes na pergunta e mudar sua ponderação lexical.
As traces registram expressões, URIs, relações e chunks das sementes.

O grafo utilizado é a visão `related_concepts` dos cartões JSONL: **54.464
arestas dirigidas resolvidas por labels únicos**. O exportador do notebook
limitou esse campo a 25 vizinhos por entidade; portanto não é o conjunto completo
de arestas do Turtle. Não foi introduzido um parser Turtle parcial nem uma nova
dependência. Os 8.342 cartões têm definição vazia; 680 entidades têm menções
identificadas no corpus. Houve sementes e expansão em 13 das 22 perguntas.

## Resultados

| Variante | nDCG documental@10 | Recall grau 2 no contexto completo | Recall graus 1–2 | Ocorrências de chunks selecionados | De documentos grau 2 | De documentos grau 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controle | 0,8816 | 0,9773 | 0,9784 | 1162 | 602 | 413 |
| Aliases | 0,8812 | 0,9773 | 0,9632 | 1155 | 595 | 418 |
| Uma aresta | 0,8453 | 0,9773 | 0,9556 | 1123 | 580 | 402 |

O controle reproduziu exatamente os contextos da run anterior com a mesma
configuração. A primeira política de aliases não melhorou nDCG e perdeu parte
da cobertura de grau 1. A expansão por uma aresta regrediu em nDCG. Nenhum
resultado justifica substituir o híbrido atual nesta etapa.

Os totais contam **ocorrências por pergunta**, não chunks únicos. Grau documental
é um proxy: um chunk de artigo grau 2 pode ser irrelevante para a pergunta,
inclusive se estiver nas referências. Grau 0 também é uma anotação provisória
da IA, não uma garantia de ausência de suporte.

## Seções: o que foi confirmado

O Docling instalado possui `SectionHeaderItem`, com texto e nível. Em
`docling_core/transforms/serializer/markdown.py`, `_format_heading` exporta
esses itens como títulos Markdown. Isso não significa que nossa lista de
strings de chunks possua um atributo de seção: o pipeline original o descartou.

`src/chunk_diagnostics.py` reconstrói o Markdown normalizado a partir dos chunks,
verificando cada overlap de 150 caracteres. Isso permite herdar o título mesmo
quando ele não aparece no próprio chunk. A seção primária é a de maior extensão
de caracteres; títulos e extensões de todas as seções atravessadas são mantidos,
assim como a marcação de chunks mistos. Os textos não são rechunkados.

O tipo de seção é uma **heurística sobre o título**, não uma classificação
semântica produzida pelo Docling. Títulos específicos ficam em `other`; antes
de um título identificado, a seção é `unknown`. A classificação não distingue
automaticamente todos os subtítulos pelo contexto hierárquico. Para um eventual
chunk truncado, a anotação continua sendo a do chunk original, com a truncagem
explicitamente marcada.

| Seção primária | Controle | Aliases | Uma aresta |
| --- | ---: | ---: | ---: |
| Introdução/background | 121 | 131 | 140 |
| Abstract/resumo | 15 | 23 | 23 |
| Métodos | 72 | 71 | 65 |
| Resultados | 66 | 64 | 58 |
| Conclusão | 52 | 46 | 47 |
| Referências | 23 | 31 | 32 |
| Outros títulos | 803 | 778 | 747 |

A tabela não inclui as categorias menores de discussão e agradecimentos;
`section-grade-counts.csv` contém o cruzamento completo seção × grau documental,
e `diagnostics.json` contém os valores por pergunta, frações e chunks mistos.
A categoria `other` é majoritária: não tratar essa primeira taxonomia como
rotulação precisa de todas as seções. Referências não foram filtradas nem
penalizadas; essas contagens permitem formular um experimento posterior.

## Validação e recursos

Passaram 28 testes, incluindo ligação com fronteiras de tokens, aliases
ambíguos, ausência de expansão recursiva, reconstrução de seções e rejeição de
overlap inválido. A execução não carrega o LLM nem reprocessa PDFs; seus tempos
não medem a execução total da submissão. Manifestos registram hashes dos dados
e código. Não houve alteração no `main.py` ou no ZIP enviado.

Não foram adicionados modelos ou dependências. Os arquivos de pesquisa não
entram no pacote atual; integrar KG na inferência futuramente exigirá incluí-los
explicitamente, medir o ZIP completo e a carga do recurso. A margem dos 60
minutos deve ser medida com o pipeline final na H100. Não há resultado novo de
acurácia de respostas nem demonstração de ganho no benchmark oculto.
