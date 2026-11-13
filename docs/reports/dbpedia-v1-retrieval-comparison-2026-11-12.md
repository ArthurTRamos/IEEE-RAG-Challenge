# DBpedia v1: auditoria e comparação de recuperação — 12/11/2026

## Parecer

O bundle `experiments/runs/dbpedia-integrity-aliases-v1` passou na validação
estrutural implementada: hashes, tamanhos, contagens, formato do subconjunto RDF
emitido, índice de aliases e correspondência de labels/arestas com o Turtle.
Isso não certifica equivalência semântica dos redirects nem completude absoluta
de um endpoint vivo. A semente ausente está explicitamente registrada.

Foi executada a comparação controlada com o recurso anterior. **Não há ganho
que justifique substituir o híbrido atual:** o recurso novo reduziu o nDCG das
duas políticas de expansão. A integridade do empacotamento melhorou; a seleção
de aliases para consulta continua frágil, e foi identificado um defeito de
extração de nomes de URIs com barra interna. Não promover este bundle à submissão.

## Auditoria do recurso

| Medida | Anterior | Novo |
| --- | ---: | ---: |
| Entidades | 8.342 | 8.343 |
| Entidades com definição | 0 | 0 |
| Entidades com propriedades | 101 | 101 |
| Propriedades de exibição | 151 | 151 |
| Entidades com aliases | 5.416 | 5.635 |
| Aliases | 22.996 | 27.436 |
| Nomes ambíguos por `casefold` | 87 | 104 |
| Arestas resolvidas dos cartões | 54.464 | 60.021 |
| Entidades com menções nos chunks | 680 | 692 |
| Perguntas com expansão | 13/22 | 13/22 |

O novo grafo contém 79.414 triplas. Não removeu nenhuma URI antiga; a única
adicionada é `http://dbpedia.org/resource/Demo`. Das 65 sementes solicitadas,
64 têm metadados e uma, `Region_of_practical_equivalence`, está marcada como
`excluded_missing_label`, com justificativa. É a única URI listada como ausente
no inventário de metadados.

As 60.021 arestas estruturadas são também as 60.021 resolvidas pelo recuperador
atual: não houve perda por labels primários ambíguos nesta execução. O consumidor
experimental ainda usa `related_concepts`, não `edges.jsonl`.

Há 148 cartões com múltiplos registros de label e 110 labels primários mudaram
em relação ao artefato anterior. Portanto, esta comparação troca um recurso
inteiro: inclui paginação, aliases completos, escolha determinística de labels
e arestas sem corte. Não atribuir toda mudança exclusivamente à paginação.

Continuam ausentes os nomes exatos `OTR`, `DONES`, `IFMIF-DONES`, `AGI` e `FCM`.
A maior cobertura de aliases não resolveu essas lacunas e não acrescentou
perguntas cobertas pelo vinculador. A pergunta 1, sobre OTR/scintillation,
continua sem sementes e sem expansão.

## Ressalvas semânticas e defeito encontrado

- `DEMO → Demo`: o cartão de destino reúne “Demo (disambiguation)”, “Demo
  version” e “Demonstration Power Plant”. O redirect existe no recurso coletado,
  mas não identifica inequivocamente o conceito nuclear desejado.
- `Lithium-6` e `Lithium-7` apontam para `Isotopes_of_lithium`; `Beryllium-7`
  aponta para `Isotopes_of_beryllium`; `Unit_ball` aponta para `Unit_sphere`.
  Esses caminhos precisam de revisão conceitual antes de serem tratados como
  equivalência. O inventário contém também nós específicos, aumentando colisões.
- O alias `2` de `Half-life` vem de `http://dbpedia.org/resource/T1/2`.
  `name_from_uri` corta na última barra e perde `T1/`. É um defeito do construtor,
  agora confirmado pelos dados. A correção deve remover o prefixo completo
  `/resource/`, preservando barras internas; não basta filtrar aliases numéricos.
- O recuperador escolhe os dois primeiros aliases em ordem alfabética. Com mais
  nomes disponíveis, códigos pouco úteis passam à frente de nomes descritivos.
  Ter procedência no arquivo não resolve isso enquanto o consumidor não a usa.

Esta auditoria não modifica o bundle nem corrige retroativamente os resultados.

## Desenho da comparação

Foram executadas duas novas runs, com recurso antigo e novo. Ambas reutilizam
os mesmos 5.174 chunks, scores densos e julgamentos das 22 perguntas ativas.
Configuração constante: BM25 k1=1,25/b=0,75, RRF 60, pesos denso/BM25 1/0,75,
contexto de 10.000 tokens, até oito sementes, dois aliases por semente e oito
vizinhos. Não houve geração por LLM, reprocessamento dos PDFs ou novo encoder.

As configurações, manifesto da fonte, hash do ranking denso e julgamentos foram
conferidos. O controle reproduziu os 22 contextos e todos os rankings
avaliados. A repetição com o recurso antigo reproduziu exatamente todos os
campos dos resumos publicados anteriormente, exceto tempos.

## Resultados

Recall refere-se a documentos no contexto completo, com os julgamentos locais
provisórios. As contagens de chunks são ocorrências por pergunta.

| Variante | nDCG documental@10 | Recall graus 1–2 | Recall grau 2 | Chunks | De docs grau 2 | De docs grau 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controle híbrido | 0,8816 | 0,9784 | 0,9773 | 1.162 | 602 | 413 |
| Aliases antigos | 0,8812 | 0,9632 | 0,9773 | 1.155 | 595 | 418 |
| Aliases novos | 0,8742 | 0,9632 | 0,9773 | 1.155 | 595 | 417 |
| Uma aresta antiga | 0,8453 | 0,9556 | 0,9773 | 1.123 | 580 | 402 |
| Uma aresta nova | 0,8384 | 0,9697 | 0,9773 | 1.131 | 576 | 411 |

Aliases novos perdem 0,0070 de nDCG frente aos antigos e 0,0073 frente ao
controle. Uma aresta nova perde 0,0069 frente à antiga e 0,0432 frente ao
controle. O recall graus 1–2 de uma aresta melhora frente à variante antiga,
mas a ordenação piora e o controle continua superior. Não há diferença no
recall grau 2 no contexto completo.

### Por domínio

| Variante | DONES (10 perguntas) | IA (12 perguntas) |
| --- | ---: | ---: |
| Controle | 0,8042 | 0,9461 |
| Aliases antigos | 0,8230 | 0,9298 |
| Aliases novos | 0,8076 | 0,9298 |
| Uma aresta antiga | 0,7808 | 0,8990 |
| Uma aresta nova | 0,7656 | 0,8990 |

A regressão nova agregada concentra-se em DONES. Em IA o nDCG é igual ao da
variante antiga, embora alguns chunks e ordens internas tenham mudado.

### Recuperações por pergunta

**Aliases, novo versus antigo:** 0 vitórias, 1 derrota e 21 empates em nDCG.
Os contextos textuais são idênticos em 19/22 perguntas; mudaram nas perguntas
3, 7 e 23. Na pergunta 3 o conjunto de chunks é igual, mas o contexto ordenado
muda. Na 23 há troca de chunks apesar do nDCG documental igual.

A pergunta **7**, sobre temperatura do nitrogen trap, explica toda a perda de
nDCG da política de aliases frente ao recurso antigo: **0,8015 → 0,6473**.
No controle, a pontuação é 0,9476. Os nomes adicionados para Nitrogen mudaram
exclusivamente de “Allotropes of nitrogen”/“Atmospheric nitrogen” para
“ATC code V03AN04”/“ATCvet code QV03AN04”; as demais adições dessa pergunta
permaneceram iguais.

No ranking documental, DONES-001 passa da primeira para a segunda posição
frente aos aliases antigos e DONES-015 sobe para primeiro. No contexto novo,
o primeiro chunk é `DONES-015.pdf#6`, uma introdução sobre HFTM, enquanto o
controle começa por `DONES-001.pdf#11`, da seção de controle de impurezas.
Frente aos aliases antigos, saem cinco chunks e entram seis; entre os removidos
estão `DONES-034.pdf#73` (hot trap/nitrogênio) e `DONES-035.pdf#72` (estratégia
do nitrogen trap). A troca não demonstra perda da resposta inteira, mas revela
deslocamento de trechos potencialmente úteis por conteúdo geral.

Na pergunta **3**, o alias numérico `2` aparece na expansão de Half-life.
O nDCG não muda, mas a trace evidencia ruído evitável. Na pergunta **23**,
“Federated stochastic gradient descent” é uma adição nova, com troca de chunks,
sem ganho no nDCG documental.

**Uma aresta, novo versus antigo:** 3 vitórias, 1 derrota e 18 empates; contextos
idênticos em 13/22 perguntas. As melhorias individuais são pequenas e a queda
na pergunta 7 é maior que sua soma:

| Pergunta | nDCG antigo | nDCG novo | Diferença |
| --- | ---: | ---: | ---: |
| 4 | 0,9328 | 0,9448 | +0,0121 |
| 5 | 0,8959 | 0,8992 | +0,0033 |
| 7 | 0,8771 | 0,7012 | −0,1759 |
| 9 | 0,4539 | 0,4624 | +0,0084 |

Os valores completos, documentos no top 10, adições de consulta, IDs e
excertos dos chunks que entram/saem estão no
[comparison.json](../../experiments/results/public22-kg-v1-2026-11-12/comparison.json)
e no [CSV por pergunta](../../experiments/results/public22-kg-v1-2026-11-12/per-question.csv).

Como limite já existente, na pergunta 22 (SimCLR), uma aresta continua com
nDCG 0,6309 contra 1,0000 do controle; o novo construtor não resolveu essa
regressão da política anterior.

### Seções do contexto novo

| Seção primária | Controle | Aliases novos | Uma aresta nova |
| --- | ---: | ---: | ---: |
| Introdução/background | 121 | 134 | 141 |
| Resumo | 15 | 24 | 22 |
| Métodos | 72 | 69 | 65 |
| Resultados | 66 | 64 | 61 |
| Conclusão | 52 | 46 | 48 |
| Referências | 23 | 31 | 32 |

Não houve filtragem por seção. A taxonomia é uma heurística baseada nos títulos,
e `other` continua majoritária. Grau documental não é julgamento de suporte de
cada passagem; esses totais não comprovam qualidade de resposta.

## Recursos e competição

O novo bundle ocupa **33.349.587 bytes (31,80 MiB)** incluindo manifesto, contra
6.470.437 bytes dos dois arquivos antigos. A comparação inclui agora arestas e
índice de aliases redundantes com os cartões, além de N-Triples mais verboso;
não é uma comparação de formatos de armazenamento equivalentes.

As runs locais levaram 15,49 s (antigo) e 15,54 s (novo), com pico de RSS de
aproximadamente 1,09 e 1,11 GiB, respectivamente. Foram executadas concorrentemente;
esses tempos são descritivos, não benchmark isolado de latência. Incluem carga,
tokenização e avaliação das três variantes, mas excluem coleta, parsing dos
PDFs, geração de embeddings e LLM. Não demonstram cumprimento do limite total
de 60 minutos na H100. O bundle não registra tempo de construção, portanto não
é possível comparar a duração da coleta a partir dele.

Não houve instalação de dependências, acesso ao endpoint ou alteração da
submissão. A avaliação continua sem internet; qualquer futuro uso exige empacotar
o recurso necessário, medir o ZIP inteiro contra 1 GB e medir execução completa.

## Reprodução e artefatos

```bash
.venv/bin/python tools/compare_kg.py --kg data/kg_rag_chunks.jsonl \
  --output-dir experiments/runs/public22-kg-legacy-replay-2026-11-12
.venv/bin/python tools/compare_kg.py \
  --kg experiments/runs/dbpedia-integrity-aliases-v1/kg_rag_chunks.jsonl \
  --output-dir experiments/runs/public22-kg-v1-2026-11-12
.venv/bin/python tools/report_dbpedia_comparison.py \
  --bundle experiments/runs/dbpedia-integrity-aliases-v1 \
  --old-run experiments/runs/public22-kg-legacy-replay-2026-11-12 \
  --new-run experiments/runs/public22-kg-v1-2026-11-12 \
  --output experiments/results/public22-kg-v1-2026-11-12/comparison.json
```

Os caminhos já existem: para repetir, escolher novas saídas. O comparador confere
igualdade de configuração, fonte, scores densos, julgamentos e controle. Os
[resumos novos](../../experiments/results/public22-kg-v1-2026-11-12/new/summary.json),
[resumos antigos repetidos](../../experiments/results/public22-kg-v1-2026-11-12/legacy-replay/summary.json),
manifestos, métricas e traces compactas foram preservados em `experiments/results/`.
Rankings completos e contextos permanecem nas runs locais, ignoradas pelo Git.
A única mudança no avaliador existente foi corrigir a descrição da limitação
sobre cortes de vizinhos; não houve mudança na política de recuperação.

## Próxima decisão

Preservar os ganhos de integridade, mas corrigir `name_from_uri`, revisar redirects
ambíguos e substituir a escolha alfabética de aliases por seleção contextual com
abstenção. Depois repetir primeiro a ablação de aliases. A disponibilidade de
mais nomes e relações, sozinha, não melhorou a recuperação local. Os resultados
não medem acurácia de respostas nem permitem concluir sobre o benchmark oculto.

A implementação e os resultados dessa decisão estão em
[Decisão sobre aliases DBpedia](dbpedia-alias-decision-2026-11-12.md).
