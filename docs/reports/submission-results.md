# Resultados de submissões

Registro iniciado em 5/11/2026 com valores informados pelo usuário. Os resultados
não foram reproduzidos localmente. IDs e tempos oficiais ainda não foram
fornecidos. Os registros novos identificam o artefato local quando disponível;
os hashes dos históricos continuam desconhecidos.

| ID local | Variante | Acurácia informada | Origem/estado |
| --- | --- | ---: | --- |
| HIST-01 | RAG com recuperação semântica | 57 | Informado pelo usuário |
| HIST-02 | RAG semântico + BM25 | 47 | Informado pelo usuário |
| HIST-03 | RAG semântico + BM25 + KG por menções/aliases | 57 | Informado pelo usuário |
| NEXT-01 | RAG semântico + BM25 + expansão do grafo | 53 | Informado pelo usuário |
| EMB-01 | Híbrido + BGE-small, top 10 sem expansão, sem KG e sem fórmulas | 40 | Informado pelo usuário em 08/11; artefato local identificado abaixo |
| DENSE-01 | MiniLM denso, top 10, chunking por seção, penalizações e prompt atual | 41 | Informado pelo usuário em 10/11; associado ao ZIP denso pelo contexto da conversa |

Os valores estão preservados na escala relatada. Não convertê-los em número
de perguntas corretas: a avaliação oficial é balanceada por domínio e o painel
público usa apenas parte do benchmark.

Um empate agregado não demonstra identidade dos acertos. As versões precisam
ser associadas a arquivos/configurações antes de atribuir uma diferença apenas
ao canal de recuperação.

Próximos passos e critérios de comparação:
[plano de experimentos](../plans/experimentation-roadmap.md).

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
