# Resultados de submissões

Registro iniciado em 5/11/2026 com valores informados pelo usuário. Os resultados
não foram reproduzidos localmente nem conferidos diretamente no painel oficial.
IDs oficiais ainda não foram fornecidos. Em 11/11/2026 foram informados os
tempos de inferência de três variantes. Os registros novos identificam o artefato local quando disponível;
os hashes dos históricos continuam desconhecidos.

| ID local | Variante | Acurácia informada | Tempo de inferência (s) | Origem/estado |
| --- | --- | ---: | ---: | --- |
| ZERO-01 | Zero-shot baseline | 29 | 732 | Acurácia balanceada por domínio; informado em 11/11 |
| RAG-BASE-01 | RAG semântico baseline | 57 | 2079 | Acurácia balanceada por domínio; informado em 11/11 |
| HYBRID-2026-11-10 | MiniLM + BM25 k1=1,25, b=0,75; RRF 60, pesos 1/0,75 | 60 | 2389 | Acurácia balanceada por domínio; informado em 11/11; manifesto identificado |
| DBPEDIA-LEGACY-2026-11-12 | Híbrido + grafo original + aliases, política legacy | 57 | — | Informado em 12/11; ZIP e manifesto identificados |
| DBPEDIA-V2-CONTEXTUAL-2026-11-13 | Híbrido + grafo V2 revisado + aliases contextuais | 60 | — | Informado em 13/11; empate agregado com o híbrido; manifesto identificado |
| HIST-01 | RAG com recuperação semântica | 57 | — | Histórico informado pelo usuário; não associado automaticamente a RAG-BASE-01 |
| HIST-02 | RAG semântico + BM25 | 47 | — | Informado pelo usuário |
| HIST-03 | RAG semântico + BM25 + KG por menções/aliases | 57 | — | Informado pelo usuário |
| NEXT-01 | RAG semântico + BM25 + expansão do grafo | 53 | — | Informado pelo usuário |
| EMB-01 | Híbrido + BGE-small, top 10 sem expansão, sem KG e sem fórmulas | 40 | — | Informado pelo usuário em 08/11; artefato local identificado abaixo |
| DENSE-01 | MiniLM denso, top 10, chunking por seção, penalizações e prompt atual | 41 | — | Informado pelo usuário em 10/11; associado ao ZIP denso pelo contexto da conversa |

Os valores estão preservados na escala relatada. Não convertê-los em número
de perguntas corretas: a avaliação oficial é balanceada por domínio e o painel
público usa apenas parte do benchmark.

Um empate agregado não demonstra identidade dos acertos. As versões precisam
ser associadas a arquivos/configurações antes de atribuir uma diferença apenas
ao canal de recuperação.

## Atualização de 11/11/2026 — híbrido com score 60

O usuário informou os três resultados no ambiente de avaliação, identificando a
métrica como acurácia balanceada dos domínios e os tempos como tempo de
inferência em segundos. O registro estruturado está em
[evaluation-reported-2026-11-11.json](../../experiments/results/evaluation-reported-2026-11-11.json).

O híbrido corresponde ao
[manifesto do pacote de 10/11](../../experiments/submissions/2026-11-10-hybrid-k1_1p25-b0p75-rrf60-dw1-bw0p75.json):

- ZIP: `2026-11-10-hybrid-k1_1p25-b0p75-rrf60-dw1-bw0p75.zip`.
- SHA-256: `6fbafb07d918527a4b0af3625735c46ff7ba58576f1ec84b1471892f701e9b97`.
- Tamanho: 398.592.372 bytes.
- Configuração: BM25 `k1=1.25`, `b=0.75`; RRF 60, pesos denso/BM25 de 1/0.75;
  orçamento de contexto de 10.000 tokens de texto dos chunks.
- Resultado informado: score 60, tempo de inferência 2389 s (39 min 49 s).
- ID da avaliação, data exata de execução, scores por domínio, respostas e logs:
  não fornecidos. A data de 11/11 é a data do relato, não uma data de execução
  presumida.

Em relação ao baseline semântico, o score passou de 57 para 60: **+3 pontos na
escala relatada**, com **+310 s** de inferência (**+14,91%**). Em relação ao
zero-shot, o híbrido teve +31 pontos e +1657 s. É o maior score registrado neste
histórico, com associação explícita ao pacote local.

Os 2389 s ficam 1211 s abaixo de 3600 s. Como o tempo foi descrito como
**inferência**, ainda não está confirmado se inclui pré-processamento,
carregamento dos modelos e construção de índices; portanto esse número sozinho
não comprova a margem total de execução de 60 minutos.

Os resultados agregados não demonstram significância estatística nem isolam o
efeito da fusão sem respostas por pergunta e identificação exata dos artefatos
dos baselines. A mesma pontuação 57 em HIST-01 não basta para concluir que se
trata da mesma submissão que RAG-BASE-01. Os demais híbridos da grade local não
foram avaliados oficialmente com base nesse relato.

## EMB-01 — BGE-small: resultado oficial informado de 40

- Arquivo local: `hybrid_noKG_BGESmall.zip`, renomeado após a construção.
- Tamanho: 393.272.049 bytes.
- SHA-256: `f522fc676ae97f2a9574b45db1e552491649ed00a01acbd87da2579c596b2f4e`.
- Modelo: `BAAI/bge-small-en-v1.5`, revisão
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, limite de 512 tokens.
- Híbrido com pesos iguais, stopwords/stemming, chunking por seção,
  penalizações e top 10 sem expansão. Prompt atual; fórmulas e KG desativados.
- ID da submissão, duração e respostas/logs oficiais: não informados.

O hash da cópia renomeada foi conferido e coincide com o
[manifesto da candidata](../../experiments/runs/embedding-comparison-2026-11-08/submission.json).
A associação entre esse artefato e o score segue o relato do usuário, sem
consulta ao painel oficial.

**Conclusão atual:** esta candidata regrediu em relação aos melhores scores
históricos informados. O ganho local de nDCG documental, de 0,8220 para 0,8371
em cinco perguntas, não se confirmou em acurácia oficial. A redução de
truncamento também não demonstrou ganho de respostas. Já havia perda de
evidência importante na questão 7, apesar de o artigo continuar no ranking.

O resultado de 40 não isola o efeito do encoder em relação à versão de 57:
os históricos não fixam o mesmo chunking, prompt e seleção de contexto. O BGE
deixa de ser a opção preferida com a evidência disponível. O próximo teste
controlado recomendado é MiniLM de 256 tokens na mesma pipeline desta candidata,
mantendo o restante fixo. Isso testa a troca de encoder; não reproduz
automaticamente a versão histórica de 57.

Priorizar a ampliação da avaliação por passagens e perguntas antes de escolher
novos modelos por métricas de documento. Resultados e configuração local da
candidata estão no [relatório da comparação](embedding-comparison-2026-11-08.md).

## DENSE-01 — MiniLM denso: resultado oficial informado de 41

- Arquivo local: `dense_minilm_submission.zip`, 398.610.151 bytes.
- SHA-256: `7cac3b4871e4da5bbc10e9330792e636effd950a96a052f2b0560024b77ba76a`.
- [Registro do pacote](../../experiments/submissions/dense-minilm-2026-11-09.json).
- BM25, KG, continuidade por vizinhos e reconhecimento de fórmulas desativados.
- Chunking por seção, corte em dez chunks, penalizações e prompt atual preservados.
- ID, tempo, respostas e logs oficiais ainda não fornecidos.

O pacote usa somente o canal denso, mas preserva mudanças em relação ao starter
kit. A auditoria encontrou evidências relevantes abaixo do corte de dez chunks
e mudanças importantes de prompt e chunking. O resultado não demonstra uma
regressão causada apenas pelo encoder ou pela correção de metadados.
Veja o [diagnóstico detalhado](dense-versus-starter-2026-11-10.md).

## Atualização de 12/11/2026 — DBpedia original + aliases legacy: 57%

O usuário informou que a primeira candidata DBpedia atingiu 57% no ambiente
oficial, contra 60% do híbrido de referência: **−3 pontos percentuais**.
Registro estruturado em
[evaluation-reported-2026-11-12.json](../../experiments/results/evaluation-reported-2026-11-12.json).

- ZIP: `2026-11-12-dbpedia-original-legacy.zip`, 399.603.332 bytes.
- SHA-256: `a8bf22b59aec7d71609736c34abbf6020a108900960588d6371472a81d09df42`.
- [Manifesto da candidata](../../experiments/submissions/2026-11-12-dbpedia-original-legacy.json).
- Configuração: mesmo MiniLM, modelos auxiliares, chunking, prompt, geração e
  parâmetros BM25/RRF do híbrido, com expansão somente da consulta BM25 por
  labels/aliases do grafo original. Encoder e pergunta do prompt permanecem originais.
- Resultado recebido por relato do usuário, sem consulta independente ao painel.
  Tempo, ID oficial, data de execução, scores por domínio e respostas/logs não
  fornecidos; a data acima é a do relato.

O quase empate de nDCG local (0,8812 contra 0,8816 do controle) não se converteu
em ganho de acurácia oficial. O resultado é compatível com a hipótese de
expansão deslocar evidências úteis, mas a causa não pode ser confirmada sem
respostas/contextos por pergunta. O score não deve ser convertido em quantidade
de perguntas erradas nem associado automaticamente às outras versões históricas
que também obtiveram 57.

Manter o híbrido de score 60 como melhor referência registrada. Não promover
esta candidata legacy nem ampliar sua expansão com base neste resultado.
Na data desse relato, a V2 contextual ainda não tinha resultado informado.
Seu resultado posterior está registrado na atualização de 13/11 abaixo.

## Atualização de 13/11/2026 — DBpedia V2 contextual: 60%

O usuário informou que a V2 obteve **60%**, exatamente o mesmo score agregado
relatado para o híbrido sem KG: **0 ponto percentual de diferença**, e +3 pontos
percentuais frente à legacy (57%).

- ZIP: `2026-11-12-dbpedia-v2-contextual.zip`.
- SHA-256: `4b933399eda72513053ff4497f82ee833972a90ac36c051d1be69257dcfe9b44`.
- [Manifesto da candidata](../../experiments/submissions/2026-11-12-dbpedia-v2-contextual.json).
- [Registro estruturado](../../experiments/results/evaluation-reported-2026-11-13.json).
- Fonte: relato do usuário, sem verificação independente do painel. Tempo,
  ID oficial, data exata de execução, resultados por domínio e respostas/logs
  não fornecidos. 13/11 é a data do relato.

A seleção contextual com abstenção empatou com o híbrido na acurácia agregada,
mas não demonstrou ganho. O empate não comprova identidade dos acertos nem
igualdade de custo. Manter o híbrido sem KG como referência, pois a adição do
recurso e do vinculador ainda não mostrou benefício de acurácia. Os resultados
das duas candidatas não encerram a hipótese futura de grafos ligados às
passagens dos PDFs; referem-se às políticas de expansão lexical DBpedia testadas.
