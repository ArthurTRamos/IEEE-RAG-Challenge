# Avaliação dos notebooks de KG — 11/11/2026

Revisão de `notebooks/dbpedia.ipynb` e `notebooks/Untitled1.ipynb`, por leitura
integral das células e saídas salvas. As células Python sem comandos de shell
passaram no parsing AST. A expressão de extração de siglas foi exercitada
isoladamente. Não houve instalação de dependências, consulta à DBpedia ou
execução integral dos notebooks. Não foram encontrados os arquivos gerados
`.ttl` ou `.duckdb` no repositório. As saídas salvas são evidências da execução
anterior, não medições reproduzidas nesta revisão.

## Parecer

A extração offline é uma base aproveitável. A recuperação demonstra consulta a
um grafo externo, mas ainda não implementa um canal de recuperação de evidências
dos artigos por grafo. Não integrar os notebooks diretamente à submissão atual.
Preservar como referência o híbrido de score 60 e isolar um experimento de KG.

## O que cada notebook faz

| Notebook | Implementação | Avaliação |
| --- | --- | --- |
| `dbpedia.ipynb`, célula 3 | Sementes por domínio, expansão SKOS, entidades por categoria, labels/comments, propriedades, redirects, wikilinks induzidos, PageRank e exportação Turtle/JSONL | Aproveitável como construtor offline, após controles de completude, tipos e determinismo |
| `Untitled1.ipynb`, célula 1 | `HybridKGRetriever`: ingestão DuckDB, busca vetorial BGE-small com boosts lexicais, vizinhos e intermediários comuns, contexto textual | Protótipo funcional segundo as saídas, com bloqueios de ambiente e qualidade |
| `Untitled1.ipynb`, célula 2 | Novo prompt com literatura e KG | Mudança independente que deve ser isolada nos testes |
| `Untitled1.ipynb`, célula 3 | `UnifiedCompetitionRAG`: esquema para KG/documentos, aliases de siglas e consultas | Esqueleto incompleto: não contém ingestão de nós, arestas ou chunks nem fluxo completo de resposta |

## Achados prioritários

1. **Dependências de inferência não autorizadas pelo ambiente listado.**
   `duckdb`, `rdflib` e `SPARQLWrapper` não constam em
   `requirements-participant.txt` e também não foram encontrados no `.venv`
   consultado. SPARQLWrapper/RDFLib podem ser ferramentas da preparação offline;
   os imports de DuckDB/RDFLib na recuperação não são compatíveis com a lista
   atual. Não assumir que dependências transitivas estarão disponíveis.

2. **Não há conexão entidade–passagem.** Os dois recuperadores consultam entidades
   DBpedia. A busca de literatura da classe unificada é somente similaridade
   densa sobre uma tabela separada. Nenhum método constrói ligações de entidades
   com IDs de chunks ou usa caminhos para recuperar evidências dos PDFs. O
   rótulo “hybrid” da primeira classe significa cosseno + boosts, não o atual
   BM25 + denso + RRF.

3. **Definições vazias na demonstração.** As saídas de OTR/scintillation mostram
   entidades centrais sem definição e listas como “Atomic number” e “Proton
   emission”. Isso não comprova recuperação da diferença física solicitada.
   A causa das definições vazias não pode ser estabelecida sem os artefatos.
   Medir cobertura de labels, definições e propriedades antes de indexar.

4. **Wikilinks não codificam mecanismos causais.** As consultas de pontes tratam
   arestas como não direcionadas e contam sementes ligadas, sem distinguir os
   predicados. Um vizinho compartilhado não estabelece causalidade nem explica
   um procedimento. A primeira classe promete pontes por categorias, mas faz
   `JOIN kg_nodes`; o exportador não cria cartões de categorias nessa tabela,
   impedindo que essas categorias sejam retornadas como pontes.

5. **Perda de tipos e unidades.** O extrator guarda valores apenas como
   `(texto, is_uri)`, descartando datatype e idioma; reconstrói literais simples
   no Turtle. A ingestão SQL também converte objetos para strings e limpa URIs.
   Propriedades como meia-vida precisam de unidade e procedência explícitas
   antes de sustentar cálculos. Manter URI original, tipo, unidade quando
   disponível, fonte e data de extração. Não inferir unidades ausentes.

6. **Completude e reprodutibilidade frágeis.** Consultas com `LIMIT` sem paginação
   podem truncar categorias, redirects e links sem diagnóstico. Respostas
   parciais recuperadas após timeout são cacheadas como respostas normais,
   impedindo distinguir uma extração completa. Uso de `set`, lotes não ordenados
   e cortes de listas vindas de conjuntos torna conteúdo/seleção instáveis.
   A célula 2 do extrator apaga o cache a cada execução; não executá-la como
   etapa rotineira. O cache deve incluir endpoint/revisão e estado de completude.

7. **Aliases e entity linking incompletos.** `HybridKGRetriever.build_index` não
   ingere os `aliases` exportados: usa só o mapa manual. O boost por substring
   não exige fronteiras de palavra; não há limiar de confiança ou opção de
   abstenção. A classe unificada procura aliases por tokens individuais, logo
   não resolve aliases de várias palavras dessa maneira. A regex de siglas
   capturou `The Optical Transition Radiation` e `We use Optical Transition
   Radiation` como nomes completos em testes isolados, prejudicando a busca
   por label. `STARTS_WITH` também pode associar um nome a entidades diferentes.

8. **Persistência pode mascarar ingestão parcial.** Na primeira classe, basta
   haver um nó para `build_index` retornar sem conferir arestas, propriedades,
   hashes ou modelo. Falha após inserção de nós pode deixar um índice incompleto
   considerado pronto. Usar transação e marcador de conclusão/versionamento.

9. **A classe unificada não é uma otimização validada.** Não há código de carga
   das tabelas nela. Os embeddings são declarados `FLOAT[384]`; carregar o
   encoder em FP16 não demonstra redução equivalente no armazenamento SQL.
   A consulta com filtros antecipados não basta para comprovar menor custo sem
   plano e medição. A remoção de artefatos de construção deve ocorrer apenas
   após validação do pacote; `CHECKPOINT/VACUUM` não substitui medir seu tamanho.

10. **Integração muda variáveis demais.** O código introduz BGE-small em lugar do
    MiniLM, potencial uso de GPU, novo prompt, seleção por tipo de pergunta e
    literatura com `top_k=5`. A interface oficial fornece pergunta/corpus/modelo,
    não o tipo da pergunta. O limite de 3.200 caracteres do KG não é um orçamento
    de tokens compartilhado com os 10 mil do contexto atual. Somar os dois sem
    controle pode elevar prefill/latência e confundir a atribuição dos resultados.

## Tamanho e tempo

As saídas do extrator registram 261 categorias, 8.344 entidades candidatas,
60.021 wikilinks e 79.235 triplas, com Turtle de **2,93 MiB**. A ingestão posterior
registra **8.342 nós**, diferença que precisa de inventário, não de suposição.
Somente a coleta de metadados levou 58min23s e a de wikilinks 1h13min11s: a
construção pertence à preparação offline e não pode ocorrer na avaliação.

O limite de 95 MB é aplicado somente ao Turtle, não a JSONL, embeddings,
DuckDB e pesos juntos. Se a poda nunca atingir o alvo antes de `keep_ratio <=
0.1`, o laço termina sem erro explícito nem exportação JSONL atualizada.

Para 8.342 vetores com 384 dimensões em float32, apenas a matriz ocuparia
12.813.312 bytes (aproximadamente 12,22 MiB), antes de textos, arestas e índices.
Isso sugere viabilidade de um recurso compacto, mas não mede os artefatos reais.
O ZIP atual tem 398.592.372 bytes; há espaço nominal até 1 GB, sujeito a medir
o pacote completo. O tempo informado de 2389 s do híbrido não permite atribuir
1211 s de margem ao KG sem confirmar o escopo dessa medição e medir a nova execução.

## Menor experimento recomendado

1. **Offline:** preservar o construtor como ferramenta separada; exportar um
   recurso determinístico em JSONL/SQLite e arrays NumPy, com URIs, aliases,
   predicados, tipos, fontes e hashes. Remover a dependência de DuckDB/RDFLib na
   inferência. Construir embeddings offline usando o mesmo MiniLM local.
2. **No corpus fornecido:** identificar menções nas passagens e registrar
   `entidade → chunk_id`, com grau de confiança. Siglas locais devem preservar
   a passagem que sustenta a associação; evitar aliases globais ambíguos.
3. **Por pergunta:** usar entidades confiáveis e caminhos curtos para gerar
   candidatos a chunks. Tratar wikilinks como relações associativas, com limites
   de grau e expansão. Se não houver ligação confiável, retornar ao híbrido.
4. **Comparação controlada:** primeiro testar expansão por aliases sem percorrer
   o grafo; depois adicionar expansão de uma aresta e medir ganho incremental.
   Manter encoder, prompt, BM25/RRF e orçamento total constantes. Um terceiro
   canal RRF só deve ranquear chunks vinculados, não misturar diretamente scores
   de entidades com scores de passagens.
5. **Avaliar:** entidades corretas/espúrias, cobertura de fatos por passagem,
   documentos recuperados, redundância, tokens, tempo de carga/consulta e RAM.
   Usar as 22 perguntas ativas, ablações e falhas por pergunta. As métricas
   documentais existentes não medem qualidade do contexto DBpedia isolado.

Adicionar definições/propriedades externas diretamente ao prompt pode ser um
experimento separado, com pequeno orçamento de tokens e fonte explícita, sem
retirar evidência científica prioritária de forma não controlada. Nenhum ganho
de QA ou de tempo foi demonstrado pelos notebooks.
