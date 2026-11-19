# Relevância de chunks para as 22 perguntas públicas

Classificação concluída de **113.432 pares pergunta–chunk**: 22 perguntas ×
5.156 chunks de 64 documentos. Os IDs originais foram preservados; os IDs
16, 18 e 21 estão excluídos, conforme o conjunto ativo do projeto.

Dos pares, **5.705 foram revisados semanticamente pelo Codex**, em lotes,
nos documentos de grau 1 ou 2. Os outros **107.727 receberam grau 0
automaticamente**, porque seus documentos têm grau 0 para a pergunta,
conforme a regra solicitada. Não há pares pendentes.

| Grau | Significado | Pares |
| --- | --- | ---: |
| 2 | Evidência direta para pelo menos uma parte material da resposta | 285 |
| 1 | Contexto útil, mas insuficiente, ambíguo ou incompleto | 1.003 |
| 0 | Sem suporte útil ou zero herdado do documento | 112.144 |

## Arquivos

- [summary.csv](summary.csv): contagens por pergunta, incluindo método e pendências.
- [questions/](questions/): um JSON por pergunta com todos os 5.156 julgamentos,
  graus documentais, métodos e justificativas.
- [reports/](reports/): relatórios Markdown por pergunta, listando os graus 1 e 2
  com justificativas e links para os chunks originais.
- [chunks.json](chunks.json): catálogo compartilhado com IDs, documento, índice,
  caminho do `.md`, quantidade de caracteres e SHA-256.
- [qrels.tsv](qrels.tsv): todos os pares no formato TREC, sem cabeçalho:
  `question_id<TAB>0<TAB>chunk_id<TAB>grade`.
- [reviews/](reviews/): decisões de revisão em 41 documentos, cobrindo 65 pares
  pergunta–documento. Contêm os grupos de índices e a justificativa das decisões.
- [manifest.json](manifest.json): critérios, cobertura, limitações e fingerprints
  das fontes, revisões e materializador.
- [validation.json](validation.json): verificações de integridade e cobertura.

Exemplo: a pergunta 25 está em [questions/25.json](questions/25.json), com uma
visualização dos positivos em [reports/25.md](reports/25.md). O ID
`AI-022.pdf#31` corresponde ao índice 31, iniciado em zero, do documento AI-022.
O caminho exato está em `chunks.json`.

## Critério e procedência

O grau mede a utilidade do **conteúdo do chunk para a pergunta**, não a semelhança
do título nem o grau do documento. Grau 2 pode sustentar apenas uma parte
material da resposta; não exige que o trecho contenha a resposta inteira.
Grau 1 exige contexto útil; compartilhar o tema amplo não basta.
O grau de um chunk não foi limitado pelo grau 1 do documento de origem.

As referências de resposta orientaram a revisão, mas evidência contrária a uma
premissa também pode ser relevante. Figuras ausentes e fórmulas danificadas
não tiveram seu conteúdo inferido. Tabelas estatísticas sem relação com a
pergunta e bibliografias foram consideradas segundo seu conteúdo e função.
Os arquivos Markdown originais foram preservados.

O campo `method` distingue `codex_semantic_review` de
`inherited_document_zero`. Nas revisões compactas, os índices não listados
recebem a decisão zero do revisor **após a inspeção do documento**. Essa
expansão não é um classificador lexical nem uma inferência de irrelevância
por ausência de palavras-chave. Justificativas podem ser compartilhadas por
vários chunks quando o mesmo motivo se aplica a eles; o JSON expandido
registra a justificativa em cada par.

Estes são julgamentos de IA, sem validação humana independente. Os zeros
documentais foram herdados por regra e não reavaliados semanticamente.
O conjunto tem `unit: chunk`; o avaliador documental existente precisa de
adaptação para usar esses julgamentos diretamente.

## Observações da revisão

- **Pergunta 4:** nenhum chunk recebeu grau 2. O instante de troca de
  dominância de Be-7 depende da Figura 8 de DONES-027, cujo gráfico não está
  disponível no Markdown. Legendas e contexto receberam grau 1 quando úteis.
- **Pergunta 11:** o artigo OneSearch atribui o colapso da hierarquia à
  aplicação de balanceamento em todas as camadas. As justificativas evitam
  atribuir esse resultado ao objetivo VAE sem suporte do trecho.
- **Pergunta 12:** DiCE e FACE permitem restrições; isso não comprova quais
  restrições foram efetivamente configuradas na comparação de ONB-MACF.
- **Pergunta 14:** REVEL ressalva que robustez pode assumir valores negativos;
  a premissa de intervalo [0,1] para todas as métricas requer essa distinção.
- **Pergunta 25:** limitação de capacidade de um estado finito e dificuldade
  LTI com espaçamento variável são questões distintas. AI-022 contém a
  explicação específica solicitada.

## Reprodução

Na raiz do repositório:

```bash
.venv/bin/python tools/build_chunk_judgments.py --require-complete
```

O comando verifica hashes e expande as decisões já revisadas; não executa
novamente a avaliação semântica. Usa apenas a biblioteca padrão do Python,
sem GPU, rede ou novas dependências. Revisões precisam ser refeitas se as
perguntas ou os chunks mudarem. Os hashes impedem reutilização silenciosa
com essas fontes alteradas.

Estes artefatos são locais para análise da recuperação e não foram integrados
à inferência nem ao pacote de submissão. Portanto, não acrescentam custo ao
limite de 60 minutos nem tamanho à submissão atual de 1 GB.
