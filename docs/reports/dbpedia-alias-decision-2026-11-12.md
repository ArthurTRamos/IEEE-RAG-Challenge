# Decisão sobre aliases DBpedia — 12/11/2026

Implementação da próxima decisão do relatório de comparação v1: correção de
nomes de URIs, revisão de redirects e seleção contextual com abstenção,
seguida de ablação de aliases nas 22 perguntas ativas.

## Resultado e decisão

As correções de integridade foram preservadas. A nova política recupera o recall
documental do controle, mas **não melhora seu nDCG**. Manter o híbrido atual na
submissão. A política contextual permanece experimental e optativa; a política
anterior continua disponível e é o padrão do avaliador para reproduções.
Não houve ajuste dos critérios após observar estes resultados.

| Variante | nDCG documental@10 | Recall graus 1–2 no contexto | Recall grau 2 | Chunks selecionados | Perguntas expandidas |
| --- | ---: | ---: | ---: | ---: | ---: |
| Controle híbrido | **0,8816** | **0,9784** | 0,9773 | 1.162 | 0 |
| V1, aliases alfabéticos | 0,8742 | 0,9632 | 0,9773 | 1.155 | 13 |
| V2 revisada, aliases alfabéticos | 0,8742 | 0,9632 | 0,9773 | 1.155 | 13 |
| V2 revisada, aliases contextuais | 0,8671 | **0,9784** | 0,9773 | 1.159 | 4 |

A contextual perde 0,0144 de nDCG frente ao controle e 0,0071 frente à política
alfabética da V2. O ganho de recall sobre aliases antigos não implica melhora
da ordenação ou das respostas.

## O que foi implementado

### Nomes e revisão do recurso

`name_from_uri` agora remove o prefixo completo de recurso DBpedia em vez de
cortar na última barra. Assim, `T1/2` permanece `T1/2`, inclusive com `%2F`.
A revisão local corrigiu 93 registros de nomes; não apenas o caso já conhecido.

A revisão explícita em `notebooks/dbpedia-redirect-review.json` rejeita a
associação de equivalência de Beryllium-7/Lithium-6/Lithium-7 com artigos gerais
de isótopos, de Unit ball com Unit sphere e de DEMO com o destino misto Demo.
São exceções conservadoras baseadas nos dados auditados, não um classificador
semântico geral nem confirmação de todas as relações restantes.

Os nós específicos já presentes são preservados e passam a ser a URI usada no
inventário de sementes revisadas. O nó Demo é excluído, pois seus nomes misturam
desambiguação, versões demonstrativas e planta nuclear; nenhum substituto foi
inventado. O manifesto guarda também o destino observado pelo endpoint e o
motivo da revisão. Os caminhos originais das cadeias continuam disponíveis.
Oito registros de nomes associados às quatro equivalências rejeitadas entre
nós preservados foram removidos, incluindo registros duplicados por procedência.

`tools/revise_dbpedia_bundle.py` deriva a V2 sem rede a partir do bundle V1
validado. Publica por diretório novo, com hashes, validação, regras de revisão,
registro de mudanças e hash do manifesto de origem. A V1 não é sobrescrita.
A derivação lê somente o subconjunto N-Triples validado que nosso construtor
emite; não é um parser RDF genérico.

A V2 está em `experiments/runs/dbpedia-reviewed-v2`: 8.342 cartões, 60.021
arestas, 79.413 triplas, 63 sementes presentes, uma excluída por revisão e uma
por ausência de label. Tem 27.442 aliases e ocupa **33.377.382 bytes** incluindo
o manifesto. O alias `2` foi removido de Half-life e `T1/2` preservado.

O notebook carrega a revisão antes da construção e usa uma saída V2 nova.
`build()` aplica as mesmas regras antes de publicar. A política de revisão não
invalida respostas SPARQL do cache, pois não muda consultas. Ainda não foi
executada nova coleta remota: a revisão local usa a coleta V1 existente.

### Seleção contextual

`KnowledgeExpansion(..., policy='contextual')` é uma alternativa explícita à
política `legacy`. Foi limitada a controle/aliases nesta etapa, sem expandir
uma aresta. A política usa somente pergunta e corpus, sem IDs de perguntas,
domínio, respostas esperadas ou julgamentos de relevância.

As regras, fixadas antes da avaliação, são:

1. Vincular menções por sequências completas de tokens. Nomes ambíguos não
   vinculam entidades; a entidade precisa de suporte de menção no corpus.
2. Considerar label e aliases, com até oito tokens por expressão. Rejeitar
   nomes numéricos/inaptos, ambiguidades e candidatos sem menção exata no corpus.
3. Obter termos de contexto da pergunta, removendo stopwords, números e termos
   das menções que vincularam entidades. Exigir pelo menos dois termos restantes.
4. Aceitar um candidato somente se pelo menos um chunk que contém a expressão
   também contém dois termos de contexto e pelo menos 50% do contexto restante.
5. Priorizar cobertura contextual, depois quantidade de chunks de suporte;
   ordem lexical é somente desempate. Selecionar até dois candidatos por entidade.
6. Adicionar apenas tokens novos ao BM25, sem repetir os já presentes na pergunta
   ou adicionados por outro candidato. A pergunta para o encoder e para o LLM
   permanece original. Se nenhum candidato passar, abster-se da expansão.

As traces guardam candidatos, motivos de rejeição, cobertura, IDs de chunks de
suporte e tokens efetivamente acrescentados. Presença lexical em um chunk não
comprova equivalência do nome nem suporte factual: é uma heurística conservadora.
Além da seleção, a política altera a deduplicação dos termos; a comparação da
política contextual avalia esses componentes juntos, não os isola causalmente.

## Ablações e recuperações

Três runs novas foram executadas: V1 com política anterior, V2 com política
anterior e V2 contextual. Todas reutilizam os mesmos 5.174 chunks e scores densos
congelados, encoder, BM25/RRF e orçamento de 10.000 tokens. Configurações básicas,
hash do ranking denso, manifestos da fonte, julgamentos e controles foram
conferidos. Os 22 contextos do controle são idênticos entre as três runs.
A V1 reproduz o nDCG da execução anterior.

A revisão do **recurso** isoladamente empata em nDCG nas 22 perguntas e mantém
21 contextos textuais idênticos. Corrigir nomes e redirects é útil para a
qualidade dos dados, mas não mostrou ganho de ordenação nesta política.

A política **contextual versus alfabética V2** tem três vitórias, três derrotas
e 16 empates em nDCG; nove contextos são idênticos. A diferença contextual versus
controle é mais esclarecedora: 19 empates e três derrotas, com 18 contextos
idênticos. Não há vitória sobre o controle nesta amostra.

| Pergunta | Controle | V2 alfabética | V2 contextual |
| --- | ---: | ---: | ---: |
| 7 — nitrogen trap | 0,9476 | 0,6473 | 0,6782 |
| 9 — volume do edifício | 0,6590 | 0,9328 | 0,6278 |
| 25 — Mamba/LTI | 1,0000 | 0,8045 | 0,9832 |

A contextual expandiu as perguntas 7, 9, 20 e 25. Foram cinco expressões
selecionadas, com tokens deduplicados. Houve abstenção em 18 perguntas, nove
com entidades reconhecidas e nove sem entidades. Os motivos registrados entre
os candidatos incluem 163 ausências de menção no corpus, 21 ausências de suporte
contextual, 37 nomes já contidos na consulta, sete expressões inválidas e duas
ambiguidades/inaptidões. Um candidato adicional ficou redundante após outra adição.

Na pergunta 7 os códigos ATC não entram na expansão. Ainda são acrescentados
termos da forma longa de IFMIF e “temperatures”; a ordenação melhora frente à
política alfabética, mas permanece abaixo do controle. Na pergunta 9, a forma
longa de IFMIF também passa pela heurística e o ranking piora. Na 25, acrescentar
somente “invariant” preserva muito mais da ordenação do controle. Isso sugere
que suporte contextual lexical ainda permite expansão genérica demais; não
comprova que qualquer uma dessas adições represente a evidência pedida.

## Validação, recursos e limites

Passaram **49 testes**, incluindo seis novos para barras em URIs, seleção com
suporte lexical/contextual, abstenção, ambiguidades, deduplicação de termos e
preservação do recurso original durante revisão. Também foi validada a V2
publicada. Não houve instalação de dependências, GPU ou chamada a LLM.

As runs completas de comparação levaram aproximadamente 27 segundos cada,
executadas concorrentemente. Os picos de RSS foram aproximadamente 1,10 GiB para
as políticas anteriores e 1,15 GiB para a contextual. Esses valores incluem
carga e avaliação local; não são benchmark isolado da seleção. O índice de
ocorrências por expressão e os tokens por chunk acrescentam memória. Não foi
medido novo tempo de coleta, pré-processamento de PDFs, embeddings ou respostas.

Nenhum recurso foi incluído no ZIP. A V2 cabe no teto local de 95 MiB, mas isso
não valida a submissão de 1 GB ou os 60 minutos totais na H100. A execução final
continua offline; eventual integração exigirá empacotar o recurso e medir o
pipeline completo. Não há resultado novo de acurácia de QA ou benchmark oculto.

## Reprodução

```bash
python3 tools/revise_dbpedia_bundle.py \
  --source experiments/runs/dbpedia-integrity-aliases-v1 \
  --output experiments/runs/dbpedia-reviewed-v2

.venv/bin/python tools/compare_kg.py \
  --kg experiments/runs/dbpedia-reviewed-v2/kg_rag_chunks.jsonl \
  --modes control aliases --alias-policy contextual \
  --output-dir experiments/runs/public22-v2-contextual-decision-2026-11-12

.venv/bin/python tools/report_alias_decision.py \
  --v1-run experiments/runs/public22-v1-legacy-decision-2026-11-12 \
  --v2-run experiments/runs/public22-v2-legacy-decision-2026-11-12 \
  --contextual-run experiments/runs/public22-v2-contextual-decision-2026-11-12 \
  --output experiments/results/public22-alias-decision-2026-11-12
```

Para reproduzir as duas runs alfabéticas, usar `--alias-policy legacy` com o
respectivo JSONL e `--modes control aliases`. Os diretórios já existem e exigem
novos nomes para repetir. A política padrão continua sendo `legacy`.

[Comparação por pergunta, documentos e chunks](../../experiments/results/public22-alias-decision-2026-11-12/comparison.json).
As subpastas desse resultado preservam resumos, manifestos, métricas e traces.
Rankings e contextos completos estão nas runs locais. A implementação atende à
próxima decisão, mas seus resultados mantêm o híbrido como opção recomendada.
