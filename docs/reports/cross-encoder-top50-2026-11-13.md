# Cross-encoder: avaliação dos primeiros 50 chunks

Reanálise dos rankings reais de 13/11/2026, nas mesmas 22 perguntas. Não houve
nova inferência, mudança de parâmetros ou ajuste de scores: foram usados os
arquivos da comparação anterior com MiniLM-L6 e o híbrido de referência.

O corte é **50 chunks**, não 50 documentos distintos. Cada chunk recebe o grau
do documento segundo os julgamentos existentes. nDCG usa ganho `2**grau-1` e
ideal baseado no catálogo integral congelado de chunks, preservando duplicatas
de documentos. Essas métricas são proxies de relevância de passagens, não
anotações de suporte factual.

## Top-50 fixo

Médias nas 22 perguntas, com exatamente 50 chunks em todas as janelas:

| Variante | nDCG proxy@50 | Chunks grau 2 | Fração grau 2 | Chunks grau 0 | Recall documental grau 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Híbrido | 0,6156 | 26,73 | 53,45% | 16,59 | 97,73% |
| CE sobre 50 candidatos | 0,6228 | 26,73 | 53,45% | 16,59 | 97,73% |
| CE sobre 100 candidatos | 0,6332 | 27,27 | 54,55% | 15,50 | 97,73% |
| CE sobre 200 candidatos | 0,6342 | 27,36 | 54,73% | 15,36 | 96,21% |

“Grau 2” nesta tabela significa **chunk de documento grau 2**. O grau documental
não distingue uma passagem útil das referências ou de outro conteúdo incidental
do mesmo artigo. A avaliação não usa as respostas de referência para modificar
os rankings.

Reordenar somente 50 candidatos mantém exatamente o mesmo conjunto no top-50,
nas 22 perguntas. Por isso contagens, frações e recall são idênticos; nDCG pode
mudar por considerar posição. O nDCG proxy melhorou em 14 perguntas, piorou em
sete e empatou em uma.

Com 100 ou 200 candidatos, o conjunto do top-50 mudou em todas as perguntas.
Em ambos os casos, o nDCG proxy@50 melhorou em 14 perguntas e piorou em oito,
comparado ao controle. Não houve teste de significância estatística.

O CE top 100 elevou o nDCG proxy em 0,0177 e acrescentou, em média, 0,55 chunk
de documento grau 2 entre os primeiros 50, sem perda do recall grau 2 nessa
janela. O CE top 200 teve nDCG ligeiramente maior, com perda de cobertura.

## Contexto efetivo sob orçamento de tokens

O contexto continua sendo definido por 10.000 tokens de texto dos chunks,
parando no primeiro overflow. Ele **não foi alterado para um corte fixo de 50**.

| Variante | Média de chunks enviados | Mínimo–máximo | Média da fração de chunks grau 2 | Recall documental grau 2 |
| --- | ---: | ---: | ---: | ---: |
| Híbrido | 52,82 | 31–65 | 52,83% | 97,73% |
| CE sobre 50 candidatos | 52,27 | 29–65 | 52,96% | 97,73% |
| CE sobre 100 candidatos | 49,59 | 29–63 | 54,87% | 97,73% |
| CE sobre 200 candidatos | 49,18 | 29–65 | 54,21% | 94,70% |

No controle, seis contextos contêm menos de 50 chunks, quinze mais de 50 e um
exatamente 50. Portanto top-50 é uma aproximação útil para comparação, mas não
substitui avaliar o contexto que será efetivamente enviado à LLM.

As frações de contexto são médias por pergunta, usando como denominador o
número efetivo de chunks daquela pergunta. A análise fixa usa denominador 50.
As medidas de nDCG dos contextos com tamanhos variáveis ficam nos arquivos de
resultados e não devem ser confundidas com nDCG@50.

## Interpretação

Esta análise mostra um sinal favorável ao **CE sobre 100 candidatos** que não
ficava evidente apenas no nDCG documental@10: melhor composição pelo proxy no
top-50 e no contexto, mantendo o recall grau 2. Ele é uma candidata para testar
respostas finais, sem substituir automaticamente o híbrido atual.

O resultado anterior de nDCG documental@10 permanece: híbrido 0,8816 contra
0,8793 no CE top 100. Trata-se de outra unidade/corte, sem contradição com a
melhora dos proxies no top-50. Não foi demonstrado ganho em acurácia de QA.

## Reprodução e registros

```bash
.venv/bin/python tools/evaluate_chunk_windows.py \
  --source experiments/runs/public22-ce-minilm-l6-2026-11-13 \
  --k 50 \
  --output-dir experiments/results/public22-ce-minilm-l6-top50-2026-11-13
```

Usar outra pasta para repetir a execução. A ferramenta aceita outros valores de
`--k` e verifica perguntas, corpus e documentos julgados. Os resultados são
gravados em [summary.json](../../experiments/results/public22-ce-minilm-l6-top50-2026-11-13/summary.json),
[topk-summary.csv](../../experiments/results/public22-ce-minilm-l6-top50-2026-11-13/topk-summary.csv)
e [per-question.json](../../experiments/results/public22-ce-minilm-l6-top50-2026-11-13/per-question.json).
O manifesto contém hashes dos rankings/contextos de origem, julgamentos e código.

Não foram alterados pipeline, modelos, pacote de submissão ou uso de recursos
na avaliação oficial. Esta é uma análise dos artefatos locais, sem dependências
adicionais, sem impacto no ZIP de 1 GB e sem nova medição do limite de 60 minutos.
