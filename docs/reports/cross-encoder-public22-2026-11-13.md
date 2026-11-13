# Cross-encoder MiniLM-L6: avaliação inicial

Execução de 13/11/2026 nas 22 perguntas ativas, com os mesmos 5.174 chunks e
scores densos congelados. Sem DBpedia, alteração de prompt, encoder ou parâmetros
do híbrido: BM25 k1=1,25/b=0,75, RRF 60, pesos 1/0,75. O controle reproduziu
exatamente os contextos da execução anterior. Não houve geração pelo LLM.

## Método

Foi usado o [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
um modelo de pares pergunta–passagem disponível na biblioteca oficial
[Sentence Transformers](https://sbert.net/docs/cross_encoder/pretrained_models.html).
Uma cópia do snapshot já existente no cache foi preparada localmente, sem
instalar dependências ou baixar novos pesos. Revisão:
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Os hashes estão no manifesto.

O módulo `src/reranking.py` permite ativar o cross-encoder em `retrieve()` e
no avaliador local. O padrão permanece sem reranker. O modelo usa CPU, float32,
máximo de 512 tokens por par, batch 16 e logits sem ativação. Os primeiros
50, 100 ou 200 chunks são reordenados pelo score CE; empates preservam a ordem
híbrida. A cauda segue na ordem original, depois do prefixo. CE e RRF não são
somados, nem seus scores são comparados entre si.

A seleção usa o mesmo orçamento de 10.000 tokens de texto dos chunks e a regra
de parar no primeiro overflow. Nenhum chunk é encurtado para compor o contexto
por causa do CE: o truncamento em 512 tokens afeta somente a pontuação, e pode
ocultar evidência presente no restante da passagem. O fallback de truncamento
do primeiro chunk excessivo, próprio do baseline, continua existindo.

## Resultados

| Variante | nDCG documental@10 | nDCG proxy por chunk@10 | Média de chunks de documentos grau 2 no top 10 | Recall documental grau 2 no contexto |
| --- | ---: | ---: | ---: | ---: |
| Híbrido controle | **0,8816** | 0,7156 | **6,55** | **0,9773** |
| CE top 50 | 0,8729 | **0,7292** | 6,50 | **0,9773** |
| CE top 100 | 0,8793 | 0,7267 | 6,45 | **0,9773** |
| CE top 200 | 0,8768 | 0,7217 | 6,41 | 0,9470 |

O nDCG proxy atribui a cada chunk o grau do seu documento e usa ganho
`2**grau-1`. O ideal considera todos os chunks do corpus com esses graus,
incluindo múltiplos chunks do mesmo artigo. Não é nDCG com julgamentos de
passagens: referências ou texto incidental de um artigo relevante podem receber
grau 2 por herança. Os julgamentos originais também são estimativas da IA.

Portanto há melhora pequena nesse proxy de ordenação (máximo +0,0136), mas
nenhum ganho de nDCG documental ou de contagem grau 2 no top 10. Com top 200,
o recall de grau 2 caiu de 0,9773 para 0,9470. A ordenação por CE tampouco
estabelece suficiência das evidências nem acurácia de resposta.

Nos contextos, houve respectivamente 1162/1150/1091/1082 ocorrências de chunks.
As quantidades provenientes de documentos grau 2 foram 602/598/587/577;
provenientes de documentos grau 0, 413/413/364/368. Isso reflete também mudanças
no comprimento das passagens selecionadas; reduzir a contagem grau 0 sem
considerar orçamento e cobertura não demonstra melhoria por si só.

## Recursos

Os arquivos do reranker somam aproximadamente **91,8 MB**. A carga do modelo
levou **3,18 s**. Pontuar 4.400 pares (200 × 22) levou **184,98 s**, na CPU com
quatro threads. O replay completo levou **204,99 s**, com pico de RSS do processo
de **1,54 GiB**. Não houve uso de VRAM para o reranker.

Os scores dos 200 candidatos foram calculados uma vez e compartilhados entre os
experimentos menores. Portanto os tempos individuais de top 50/top 100 não
foram medidos. Houve 66/1100, 144/2200 e 275/4400 pares acima de 512 tokens nas
três configurações; não extrapolar essa latência diretamente para a H100.

Esses custos excluem parsing de PDFs, inferência densa e carregamento/geração
do LLM final. Não validam o limite de 60 minutos. Os pesos adicionais parecem
compatíveis com o orçamento de 1 GB em conjunto com o ZIP anterior de 398,6 MB,
mas não foi construído ou medido um novo pacote. O build atual e o `main.py`
da submissão não habilitam nem incluem o cross-encoder.

## Reprodução e artefatos

```bash
.venv/bin/python tools/compare_reranking.py \
  --model models/ms-marco-MiniLM-L6-v2 \
  --candidates 50 100 200 --threads 4 \
  --output-dir experiments/runs/public22-ce-minilm-l6-2026-11-13
```

Use uma pasta nova para repetir a execução existente. Rankings completos,
contextos e scores ficam nessa pasta. Os registros consolidados estão em
[summary.csv](../../experiments/results/public22-ce-minilm-l6-2026-11-13/summary.csv),
[summary.json](../../experiments/results/public22-ce-minilm-l6-2026-11-13/summary.json)
e [manifest.json](../../experiments/results/public22-ce-minilm-l6-2026-11-13/manifest.json).
Cada variante tem nome explícito, manifesto, diagnóstico por pergunta e resumo.

Passaram **29 testes**, incluindo estabilidade de empates, preservação da cauda,
rejeição de scores inválidos, limite de candidatos e ativação opcional no pipeline.
Os dados das quatro runs foram validados quanto às 22 perguntas e às contagens.

## Decisão

Manter o híbrido atual como configuração de submissão. O CE fica disponível
para experimentação; a evidência desta rodada não justifica ativá-lo por padrão.
Antes de escolher outro reranker ou calibrar seus scores com RRF, priorizar uma
amostra de passagens julgadas quanto aos fatos necessários à resposta. Não
generalizar esse resultado de um modelo e 22 perguntas para todos os rerankers.
