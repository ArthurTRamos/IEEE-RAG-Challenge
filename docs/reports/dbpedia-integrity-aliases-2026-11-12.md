# Construtor DBpedia: integridade e aliases — 12/11/2026

Implementação das prioridades 1 e 2 da avaliação anterior. O notebook agora
chama `tools/dbpedia_builder.py`; sementes, filtros e propriedades anteriores
foram preservados em `notebooks/dbpedia-config.json`. O código de inferência,
os recursos em `data/` e a política experimental de recuperação não mudaram.

## O que mudou

- Removidas a instalação automática de bibliotecas e a exclusão do cache.
  O construtor usa somente a biblioteca padrão. O cache SQLite usa tabela nova,
  com namespace por versão do código, configuração, endpoint e snapshot;
  respostas antigas sem estado de completude não são reutilizadas.
- Consultas com `SELECT DISTINCT`, ordenação por todas as variáveis projetadas
  e paginação OFFSET. Há orçamento de páginas, detecção de páginas repetidas,
  tratamento conservador de sinais de truncamento do endpoint, retries e
  divisão de lotes. Falha de uma URI interrompe a construção e gera diagnóstico.
  Estados `partial` e `failed` nunca são consumidos como `complete`.
- Sementes resolvidas por redirects antes da coleta, com cadeia, URI solicitada,
  URI canônica e status. Ciclos, múltiplos destinos, limite de saltos ou semente
  sem label impedem publicação. Ausências de metadados das demais entidades
  são listadas no manifesto, sem inventar labels.
- Labels e aliases com texto, tipo, idioma e origem em `names`. Nomes de
  redirects usam idioma desconhecido, pois o título da URI não traz language
  tag. Os campos antigos dos cartões continuam disponíveis. Removidos os cortes
  arbitrários de aliases por quantidade/comprimento; ambiguidades são mantidas
  no índice nome → conjunto de URIs, sem resolução automática.
- `reviewed_aliases` permite acrescentar associações revisadas com fonte e alvo
  existente. A lista inicial está vazia: não foram introduzidas siglas sem
  comprovação. Siglas locais aos documentos e mudanças no vinculador de
  recuperação ficam para um experimento posterior, preservando o controle.
- Todos os conjuntos, lotes, cartões e relações são ordenados. Wikilinks por URI
  são preservados em `edges.jsonl`; cartões também não truncam vizinhos. Não foi
  acrescentado enriquecimento ontológico ou expansão de categorias mais profunda.
- Publicação em um diretório novo e imutável, por rename único após validação
  em staging. O manifesto `complete` inclui hashes, tamanhos, configuração e
  inventários. Verificam-se formato dos termos RDF, contagens, aliases e
  correspondência entre labels/arestas e Turtle. O Turtle usa N-Triples, um
  subconjunto de Turtle; não exige RDFLib. Arquivos com hash incorreto são rejeitados.
- O teto de 95 MiB passa a valer para o bundle completo, incluindo manifesto.
  Excesso gera erro, sem a poda PageRank automática anterior e sem publicação
  parcial. O objetivo desta versão é medir um inventário íntegro antes de
  experimentar seleção por relevância.

## Execução

Abrir `notebooks/dbpedia.ipynb` a partir da raiz ou de `notebooks/`. Rever a
configuração e o caminho `OUTPUT` antes da célula de construção. O padrão
publica em `experiments/runs/dbpedia-integrity-aliases-v1`; uma repetição exige
outro diretório de saída, mantendo o cache. Não sobrescreve `data/`.

Para invalidar o cache, incrementar `cache_version`; para uma fonte fixa,
configurar seu `endpoint` e identificador `snapshot`. Alterar somente o nome do
snapshot não torna um endpoint vivo imutável. OFFSET ordenado não elimina
alterações do dataset entre páginas, e truncamento silencioso não sinalizado
pelo servidor não pode ser excluído. Recomenda-se snapshot fixo para reprodução
entre datas. O cache persistente preserva respostas completas até invalidação.

O consumidor experimental só deve selecionar o novo JSONL depois de verificar
`validate_bundle(OUTPUT)`. `aliases` permanece uma lista de strings, mas os
novos campos de procedência não são usados pelo vinculador atual. Os ganhos
potenciais não foram medidos; corrigir o recurso não comprova ganho de QA.

## Validação e limites

Testes com transporte SPARQL simulado cobrem cache completo/parcial, isolamento
por snapshot, erro de rede/JSON, limites do endpoint, paginação, divisão de lote,
falha individual, resolução/ciclos/ambiguidades de redirects, aliases longos e
com fonte, preservação de tipos RDF, publicação determinística, corrupção e
orçamento excedido. A suíte é executada com:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

Passaram **41 testes**, incluindo **13 novos** para o construtor.
Também foi conferido o parsing das células Python do notebook. Não houve
instalação de dependências, coleta real da DBpedia ou nova avaliação de
recuperação/QA. O comportamento com o endpoint real e as URIs antes ausentes
precisa ser confirmado na próxima coleta offline.

Esta etapa não usa GPU nem altera o ZIP. A coleta continua fora dos 60 minutos
da avaliação; seus tempos e tamanhos reais ainda não foram medidos nesta
versão. Preservar nomes e relações pode aumentar o recurso; o teto local de
95 MiB não substitui medir a submissão inteira contra 1 GB. A integração futura
exige medir carga, RAM, recuperação e execução final na H100.

## Correção após a primeira coleta real

O erro fornecido mostrou ausência de label apenas para
`Region_of_practical_equivalence` (64 das 65 sementes passaram pela verificação).
A política inicial interrompia toda a coleta, e a mensagem listava inclusive
sementes válidas. Agora a mensagem mostra somente as ausências bloqueantes.
A configuração aceita apenas essa URI em `allowed_missing_seed_labels`, com
justificativa; o manifesto registra `excluded_missing_label` e a URI permanece
no inventário de ausências. Não se cria label nem associação artificial.
Novas ausências e erros de consulta continuam bloqueando a publicação.

Essa política de publicação não muda o namespace das consultas: o cache existente
é reaproveitado. Para repetir no Jupyter, reiniciar o kernel e executar as células
com a configuração atualizada, preservando o arquivo SQLite e `cache_version`.
Não é necessário apagar o arquivo `.failed.json` da tentativa anterior; ele é
um diagnóstico separado. O novo bundle terá seu próprio manifesto validado.

O log também registra redirects como `DEMO → Demo` e
`Lithium-7 → Isotopes_of_lithium`. Esses caminhos do endpoint não comprovam
sinonímia conceitual; devem ser revisados antes de usar o novo recurso na
recuperação. Esta correção trata a ausência de label, sem validar semanticamente
esses redirects nem alegar ganho de recuperação.
