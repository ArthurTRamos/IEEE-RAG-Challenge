# Capa/metadados e bibliografia nos chunks recuperados

Análise de 14/11/2026 das quatro variantes do experimento MiniLM-L6, nas 22
perguntas ativas. Não houve nova inferência, filtragem ou mudança na submissão.

## O que é possível identificar

É possível obter **estimativas estruturais**, não uma identificação visual
exata de páginas de capa. Os rankings congelados preservam o Markdown exportado
pelo Docling, mas não a localização de cada caractere nas páginas do PDF.

A ferramenta `tools/evaluate_paratext.py` reconstrói o Markdown normalizado de
cada documento, verificando os overlaps entre chunks, e marca quatro categorias:

- `front_matter_candidate`: prefixo anterior ao primeiro marcador reconhecido
  de Abstract/Resumo ou Introduction/Introdução. Pode conter título, autores,
  afiliações, avisos de publicação e conteúdo de capa. Não equivale a uma página
  de capa confirmada e pode incluir texto científico sem cabeçalho reconhecido.
- `bibliography`: seção com cabeçalho reconhecido de referências/bibliografia,
  até o próximo título do mesmo nível ou superior. Citações no corpo não são
  classificadas como bibliografia apenas por terem referências bibliográficas.
- `body_candidate`: texto posterior ao início do corpo, fora da bibliografia;
  não implica informação relevante e pode incluir agradecimentos ou metadados.
- `unknown`: estrutura insuficiente para atribuição. Não é convertida em corpo
  por padrão. O documento DONES-011 não teve marcador de início reconhecido.

A categoria principal é a de maior extensão de caracteres do chunk, com
desempate pela primeira extensão. Também são exportadas extensões de caracteres,
fração de conteúdo de capa/bibliografia e a indicação de **chunk misto**. Um
chunk cujo conteúdo esteja dividido igualmente pode não ter maioria absoluta;
por isso a classificação principal deve ser lida junto com as extensões.

Para os contextos efetivos, o texto selecionado é conferido contra o original
e eventual truncamento é aplicado às extensões antes da contagem. A integridade
do catálogo textual é verificada contra o hash da fonte densa da execução.

## Primeiros 50 chunks — médias por pergunta

| Variante | Capa/metadados candidatos | Bibliografia | Corpo candidato | Incertos | Mistos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Híbrido | 1,41 | 1,14 | 47,36 | 0,09 | 3,09 |
| CE sobre 50 | 1,41 | 1,14 | 47,36 | 0,09 | 3,09 |
| CE sobre 100 | 2,05 | 1,14 | 46,68 | 0,14 | 3,41 |
| CE sobre 200 | 1,68 | 1,23 | 47,05 | 0,05 | 3,59 |

As quatro categorias principais somam 50 antes do arredondamento. “Mistos” é
uma marcação sobreposta e não deve ser somada a essas categorias. Os totais de
capa/bibliografia nas 22 perguntas são 31/25 no híbrido, 31/25 no CE 50, 45/25
no CE 100 e 37/27 no CE 200.

Considerando **qualquer presença**, e não apenas categoria principal, há em
média 4,55 chunks com algum conteúdo de capa/bibliografia no top-50 do híbrido,
contra 5,45 no CE 100. Essa diferença decorre sobretudo de chunks mistos.

## Contexto efetivamente selecionado — totais nas 22 perguntas

| Variante | Total | Capa/metadados candidatos | Bibliografia | Corpo candidato | Incertos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Híbrido | 1162 | 32 | 21 | 1106 | 3 |
| CE sobre 50 | 1150 | 34 | 22 | 1091 | 3 |
| CE sobre 100 | 1091 | 41 | 25 | 1022 | 3 |
| CE sobre 200 | 1082 | 33 | 21 | 1027 | 1 |

São ocorrências por pergunta, não chunks únicos no corpus. O contexto real
continua sendo limitado pelo orçamento de tokens, não por um corte fixo em 50.

## Interpretação

O reranker não reduziu sistematicamente a presença dessas categorias. CE 100,
que apresentou melhora nos proxies do top-50, elevou a presença de conteúdo
inicial candidato e manteve a média de bibliografia na janela fixa.

Isso permite formular uma ablação de penalização estrutural ou seleção de
trechos dentro de chunks mistos, mas não justifica remover automaticamente
essas passagens. Títulos podem ajudar a identificar o artigo correto; bibliografia
pode conter nomes de métodos relevantes; um chunk misto pode carregar a evidência
necessária em sua parte científica. Classificação estrutural não é julgamento
de utilidade factual.

É recomendável auditar os casos marcados antes de qualquer filtro: em especial,
os candidatos a capa e os chunks mistos de documentos grau 2. Para confirmar
capas reais, seria necessário reprocessar preservando os itens e as páginas do
Docling e alinhá-los aos chunks; essa reextração não foi feita nesta análise.

## Artefatos e reprodução

```bash
.venv/bin/python tools/evaluate_paratext.py \
  --k 50 \
  --output-dir experiments/results/public22-ce-paratext-2026-11-14-verified
```

Use uma pasta nova para repetir a execução existente. Os arquivos estão em
[summary.json](../../experiments/results/public22-ce-paratext-2026-11-14-verified/summary.json),
[per-question.csv](../../experiments/results/public22-ce-paratext-2026-11-14-verified/per-question.csv)
e [per-chunk.csv](../../experiments/results/public22-ce-paratext-2026-11-14-verified/per-chunk.csv).
O CSV por chunk inclui ID, grau documental, categorias, frações e um excerto
para inspeção. As anotações completas estão em `annotations.json`; hashes e
definições ficam no manifesto.

Passaram 37 testes, incluindo chunks mistos, transição de bibliografia para
apêndice, truncamento, conteúdo sem marcador estrutural e overlap inválido.
Não foram adicionados modelos ou dependências, nem alterados custos de
inferência, pacote de 1 GB ou comportamento sob o limite de 60 minutos.
