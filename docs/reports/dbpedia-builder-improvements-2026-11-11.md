# Melhoria do construtor de subgrafos DBpedia

Avaliação em 11/11/2026 de `notebooks/dbpedia.ipynb`, dos requisitos locais,
da revisão anterior e dos experimentos de recuperação já registrados.

## Parecer

É possível melhorar substancialmente a qualidade e a utilidade dos recursos
exportados. Ainda não há evidência de que essas melhorias aumentarão a qualidade
das respostas. A prioridade é tornar o grafo **completo dentro de um escopo
declarado, desambiguado e utilizável para recuperar passagens**. Aumentar a
profundidade das categorias ou o número de wikilinks, isoladamente, não é uma
hipótese bem sustentada pelos resultados atuais.

Recomendo manter o híbrido atual como controle e evoluir o construtor offline
em etapas. Esta avaliação não altera o notebook nem o pipeline de submissão.

## Evidências atualizadas

A [revisão anterior](dbpedia-notebooks-review.md) não tinha os artefatos locais.
Agora foi possível auditar diretamente `data/kg_rag_chunks.jsonl`. As contagens
e hashes estão em [dbpedia-builder-audit-2026-11-11.json](../../experiments/results/dbpedia-builder-audit-2026-11-11.json).
O método foi leitura do JSONL e inspeção AST da célula 3; não houve consulta ao
endpoint, parsing do Turtle, instalação de dependências ou execução da coleta.

| Verificação | Resultado observado | Consequência |
| --- | ---: | --- |
| Cartões / URIs distintas | 8.342 / 8.342 | Não há URIs duplicadas no JSONL |
| Definições não vazias | 0 | O índice textual carece de descrições semânticas |
| Entidades com propriedades | 101 (1,21%) | Constantes e atributos têm cobertura muito baixa |
| Entidades com aliases | 5.416 | Há cobertura parcial de nomes alternativos |
| Nomes ambíguos após `casefold` | 87 | Um alias não pode determinar sozinho a entidade |
| Sementes presentes | 63 de 65 | A proteção de sementes na poda não garante sua exportação |
| Cartões com 25 vizinhos | 382 | A seleção arbitrária no limite afeta parte do grafo consumido |
| Turtle / JSONL | 3.074.924 / 3.395.513 bytes | O tamanho atual não exige poda agressiva |

As duas sementes ausentes são `dbr:DEMO` e
`dbr:Region_of_practical_equivalence`. O código exige label inglês para incluir
uma entidade em `metadata`; depois, tanto a consulta de wikilinks quanto a
exportação dependem desse inventário. A ausência pode envolver URI,
redirecionamento, disponibilidade ou falha de coleta; os artefatos não permitem
atribuir uma causa específica. Resolver as sementes e registrar o resultado é
mais útil do que apenas somar um bônus ao PageRank.

Não existem labels ou aliases com correspondência exata, ignorando caixa, para
`OTR`, `DONES`, `IFMIF-DONES`, `AGI`, `FCM` ou `Scintillation`. Isso mede ausência
de nomes no recurso, não ausência dos conceitos no corpus. `Transition_radiation`
possui “Optical transition radiation”, mas não a sigla OTR;
`Scintillation_(physics)` conserva a desambiguação no label.

Um exemplo concreto de conflito é `lithium-7`, associado a
`Isotopes_of_lithium` e `Lithium-7` por labels/aliases. O construtor deve
preservar a origem desses nomes para permitir preferir o label canônico
quando apropriado, em vez de tratar todas as associações igualmente.

O [experimento local existente](kg-alias-onehop-2026-11-11.md) encontrou 680
entidades com menções no corpus, aproximadamente 8,15% dos cartões, e expansão
em 13 de 22 perguntas. Essa cobertura depende do vinculador lexical atual;
não prova que os demais conceitos sejam inúteis ou ausentes semanticamente.

| Variante já avaliada | nDCG documental@10 | Recall documental graus 1–2 no contexto |
| --- | ---: | ---: |
| Híbrido de controle | 0,8816 | 0,9784 |
| Expansão por aliases | 0,8812 | 0,9632 |
| Expansão por uma aresta | 0,8453 | 0,9556 |

Esses números são resultados anteriores, não uma nova execução. São métricas
documentais com julgamentos locais provisórios; não medem diretamente suporte
de resposta por passagem nem acurácia do LLM. A queda por uma aresta não isola
o efeito do grafo: inclui também a política de seleção e expansão lexical.

## Alterações prioritárias no notebook

### 1. Completude e exportação verificável — prioridade imediata

- Substituir a remoção rotineira do cache da célula 2 por invalidação explícita
  por versão. Incluir endpoint, versão do extrator, configuração e identidade
  do snapshot, quando disponível, nas chaves/metadados do cache.
- Distinguir `complete`, `partial` e `failed`. Respostas recuperadas de S1TAT
  podem auxiliar diagnóstico, mas não satisfazer uma consulta completa nem
  impedir a divisão adaptativa do lote. Registrar URIs que falham isoladamente.
- Paginar consultas com limites e ordenação estável, preferencialmente por
  chave. Reduzir lotes não resolve truncamento silencioso de uma única URI.
  Registrar limites explícitos de coleta e grupos incompletos. A paginação do
  endpoint vivo melhora a coleta, mas um snapshot fixado é necessário para
  reprodução estrita entre datas.
- Ordenar URIs, resultados deduplicados e desempates antes de fazer lotes,
  selecionar vizinhos e exportar. `list(set(...))[:N]` não é seleção estável
  nem seleção por relevância.
- Validar os dois formatos e publicar juntos, com manifesto, hashes e marcador
  de conclusão. Falhar explicitamente se o orçamento não puder ser atendido;
  impedir que um JSONL antigo acompanhe um Turtle novo.

Essas correções tornam os experimentos interpretáveis. Não constituem, por si
só, evidência de ganho de recuperação.

### 2. Resolver entidades e recuperar nomes úteis — alto potencial

Resolver redirects das próprias sementes antes da expansão, guardando
`requested_uri`, `canonical_uri`, cadeia de resolução e status. Hoje o notebook
apenas busca redirects que apontam para as entidades, para obter aliases; isso
não resolve uma semente que seja um redirect.

Exportar nomes como registros com texto, idioma, tipo e origem: label,
redirect, forma alternativa ou associação local do corpus. Preservar todos os
aliases válidos no recurso mestre; aplicar limites por relevância no consumo,
sem o corte arbitrário atual de 15 nomes ou o descarte automático de nomes
com mais de 35 caracteres. Guardar o índice alias → conjunto de URIs.

Siglas ausentes podem ser acrescentadas por uma lista pequena e revisada, com
fonte, ou por padrões explícitos de definição nos documentos. Não inferir que
DONES e IFMIF são sinônimos completos. Associações extraídas dos artigos devem
ser locais ao documento, guardar `document_id`, `chunk_id` e trecho de suporte.
Nomes sem o sufixo de desambiguação, como “Scintillation”, são candidatos
contextuais; não devem se tornar aliases globais inequívocos automaticamente.

Na recuperação, testar primeiro esses nomes sem acrescentar vizinhos. Evitar
repetir termos já presentes na pergunta apenas por diferenças de caixa ou
por exportar uma expressão equivalente, pois isso altera o peso no BM25.

### 3. Recuperar descrições e propriedades com procedência

A consulta atual usa somente `rdfs:comment`. Investigar por amostragem se há
`dbo:abstract` e outros campos descritivos no endpoint/snapshot utilizado,
mantendo fallback explícito entre campos disponíveis. Não é possível afirmar
que trocar o predicado resolverá as 8.342 definições vazias sem essa verificação.
Registrar ausência de conteúdo separadamente de falha de consulta.

Guardar o texto integral na preparação offline e derivar uma descrição curta
em fronteiras de sentença para indexação. O corte atual em 450 caracteres pode
eliminar qualificações relevantes. Comparar cards com e sem descrição usando
o mesmo encoder antes de ampliar índices ou contexto do LLM.

Nas propriedades, preservar o binding RDF: tipo de objeto, URI, valor literal,
datatype e idioma. Conservar unidade quando explicitamente disponível e
registrar ausência quando não estiver. Separar propriedades normalizadas `dbo:`
de propriedades brutas `dbp:`; não fundir silenciosamente valores conflitantes.
Não utilizar valores de meia-vida ou densidade em cálculos com base apenas em
strings sem unidade e fonte. O exemplo `Beryllium-7` atualmente não contém
propriedades; é um bom caso de auditoria, não uma garantia de correção rápida.

### 4. Preservar as arestas antes de escolher quais recuperar

Exportar uma tabela de arestas com `source_uri`, `predicate_uri`, `target_uri`,
direção e procedência. Labels devem ser atributos de exibição. Manter os
cartões atuais como visão derivada preserva compatibilidade com o experimento
existente. Não reconstruir o grafo mestre a partir de `related_concepts`.

Os 54.464 vínculos dos cartões já sofreram o corte em 25 vizinhos. Reordenar
essa visão não recupera os vínculos perdidos. O Turtle pode conservar arestas
adicionais; converter e comparar os conjuntos offline é uma etapa de validação,
usando um parser RDF adequado, antes de decidir refazer toda a coleta.

Distinguir taxonomia SKOS, associação de categoria, wikilink e relações
ontológicas selecionadas. Wikilinks podem gerar candidatos, mas não demonstram
causalidade, equivalência ou sequência de procedimento. Uma lista pequena de
predicados úteis deve ser escolhida após auditar disponibilidade e cobertura;
extrair todos os predicados indiscriminadamente aumentaria ruído e custo.

### 5. Selecionar por utilidade para os domínios

O notebook mistura DONES, IA e matemática numa expansão comum. Guardar a
procedência por semente, domínio e distância permite exportar duas visões com
URIs compartilhadas, sem duplicar entidades ou excluir matemática transversal.

Preferir um núcleo de sementes resolvidas e conceitos específicos, mais uma
camada de candidatos. Frequência de menção, especificidade e conectividade com
o núcleo podem ser sinais de seleção, com quotas por domínio. Frequência local
não deve eliminar definitivamente conceitos necessários ao corpus oculto.

O PageRank atual inclui arcos entidade → categoria, mas não incorpora os
`skos_edges` nesse cálculo. Categorias podem absorver massa, e o ranking final
considera somente entidades. Portanto, ele não equivale a relevância para a
pergunta. Primeiro comparar seleção sem poda, limites por grau e, como
experimento separado, PageRank personalizado por domínio ou pergunta.

A blacklist também é grosseira: o artefato inclui “1945–1998”, da categoria
“Nuclear warfare”, e “13 Things That Don't Make Sense”, de “Cold fusion”.
Isso ilustra possíveis desvios; não basta para rotular automaticamente todos
esses recursos como inúteis. Filtros devem ter relatório de exclusões e exceções
para instalações e organizações relevantes, e ser avaliados por amostragem.

## Conexão com a recuperação de evidências

Mesmo um subgrafo DBpedia melhor não contém necessariamente resultados novos,
parâmetros da instalação ou mecanismos descritos nos artigos. A etapa decisiva
é ligar entidades às passagens disponíveis na execução:

`pergunta → entidades confiáveis → aliases/relações limitadas → chunks vinculados`

O construtor offline deve fornecer os recursos para essa ligação. O índice
entidade → chunks é construído sobre o corpus efetivamente fornecido, com
IDs estáveis, seção, posição da menção e confiança. Menção constitui um sinal
de candidato; não comprova que a passagem responda à pergunta.

Uma evolução além da expansão BM25 atual seria um canal de candidatos de
passagens por grafo, fundido com BM25/denso por ranking. Esse é outro experimento:
não confundir sua eventual contribuição com a melhoria do arquivo DBpedia.
Abster-se quando o vínculo for ambíguo ou não houver suporte no corpus.

## Menor sequência de experimentos recomendada

1. **V1 — integridade:** cache/completude, resolução das 65 sementes,
   ordenação e exportação de arestas por URI. Manter a política de recuperação.
   Comparar inventários e reportar diferenças, sem exigir que falhas antigas
   sejam reproduzidas.
2. **V2 — aliases:** auditar siglas e colisões; comparar o controle, o recurso
   antigo e o novo com a mesma política lexical. Medir associação correta,
   abstenção, entidades/perguntas cobertas e termos espúrios adicionados.
3. **V3 — relações:** com V2 fixada, comparar nenhuma expansão e uma aresta
   limitada por domínio/especificidade. Avaliar primeiro as arestas recuperadas
   do Turtle; refazer a coleta apenas onde incompletude estiver demonstrada.
4. **V4 — descrições/canal de passagens:** testar separadamente descrições para
   desambiguação e recuperação de chunks por grafo. Manter encoder e orçamento
   de contexto e ajustar somente o componente em avaliação.

Usar as 22 perguntas ativas com IDs originais. Reportar diferenças por pergunta
e domínio, nDCG documental, Recall@k e cobertura no orçamento de contexto.
Acrescentar uma pequena anotação humana de evidências por passagem para
distinguir artigos corretos de trechos úteis. Congelar a política antes de
validar em perguntas adicionais, reduzindo sobreajuste ao conjunto público.
Só depois executar QA com o LLM e medir acurácia e execução completa.

## Viabilidade nos limites da competição

A coleta SPARQL e o enriquecimento DBpedia pertencem à preparação offline.
As saídas históricas registram horas de coleta; essa fase não cabe nos 60
minutos e não pode depender de internet na avaliação. Na execução final devem
entrar apenas recursos previamente preparados e ligação com o corpus fornecido.

Os dois arquivos atuais somam 6.470.437 bytes, aproximadamente 6,17 MiB antes
de compressão. Há viabilidade nominal para um grafo compacto, mas esses valores
não incluem índices, embeddings, pesos ou o restante da submissão. O alvo de
95 MB do Turtle é uma escolha local e não valida o limite oficial de 1 GB.
Medir o pacote completo; preferir um único formato de consumo e deixar o RDF
mestre fora dele se não for necessário à inferência.

JSONL, SQLite e estruturas Python permitem consumo sem DuckDB/RDFLib na
avaliação. Ferramentas RDF podem permanecer na preparação offline; não adicionar
dependências de inferência ausentes do `requirements-participant.txt`.
As primeiras etapas não exigem GPU ou novo modelo. A ligação com chunks deve
ser indexada e sua carga, latência e RAM medidas; não presumir custo desprezível.
O limite de 60 minutos inclui também PDFs, índices, modelos e todas as respostas,
e só pode ser validado no pipeline completo na H100.

## Decisão recomendada

Prosseguir com a melhoria do construtor como experimento offline, começando por
integridade, resolução de sementes e aliases. Aumentar a expansão taxonômica
fica em prioridade menor. A política atual de uma aresta já mostrou regressão
local; promover um novo recurso à submissão exige demonstrar ganho de evidência
ou de QA sob os mesmos orçamentos, além de medir tempo e tamanho finais.
