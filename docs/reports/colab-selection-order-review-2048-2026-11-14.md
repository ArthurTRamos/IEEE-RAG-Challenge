# Revisão das duas ordenações com saída de 2.048 tokens

## Resultado principal

As duas runs recebidas com sufixo `_answers (1).json` usam 2.048 tokens de
saída. Ambas concluíram as 22 perguntas com status `ok`, sem OOM registrado,
e **nenhuma atingiu o limite**. As conclusões antes cortadas agora podem ser
avaliadas. Isso resolve a restrição observada na revisão com 1.024 tokens,
mas não os erros de conteúdo das questões 5, 13 e 23.

O sinal favorável à seleção híbrida com ordem CE permanece, especialmente na
questão 6: mesmo concluída, a resposta em ordem híbrida termina numa estimativa
de 30–50 L/s; a ordem CE termina em 97,5 L/s, próximo dos aproximadamente
98 L/s da referência. A diferença não dependia apenas do corte em 1.024 tokens.

## Integridade e configuração

Foi repetida a auditoria de IDs, referências do `sample_qa.json`, hashes das
entradas, prompt de sistema e mensagens completas. As duas variantes recebem
os mesmos chunks e textos, em ordens diferentes. As configurações registradas
são iguais entre si. Comparadas ao registro anterior, a única diferença nos
metadados é `max_new_tokens`: 1024 → 2048. Continuam FP16/NF4, geração greedy,
cache habilitado com implementação não especificada e as mesmas versões.

Não há evidência de novo corte de entrada: os hashes de entrada permanecem
iguais aos registrados na rodada anterior. A inferência é a do 7B quantizado,
não a do OLMo 3.1 32B oficial. Mapa de dispositivos, revisão exata do modelo,
tempo e motivo de término/EOS continuam ausentes dos registros. Não atingir
o limite é verificável; término por EOS não foi registrado explicitamente.

| Medida | Híbrido–híbrido | Híbrido–CE |
| --- | ---: | ---: |
| Questões concluídas | 22 | 22 |
| Questões que atingem 2.048 tokens | 0 | 0 |
| Média de tokens gerados | 647,05 | 633,95 |
| Maior pico de VRAM alocada (GiB) | 10,82 | 10,82 |
| Maior pico reservado (GiB) | 12,18 | 12,18 |

O aumento do teto de saída não elevou materialmente o maior pico registrado
neste conjunto. Isso não garante a mesma memória para outras entradas.

Os arquivos completos anteriores de 1.024 tokens não estão mais na pasta
fornecida. A auditoria histórica preserva tamanhos e metadados, mas não hashes
por resposta. Assim, podemos verificar que só aumentaram os comprimentos das
questões antes limitadas, mas não provar igualdade textual das demais respostas
entre rodadas. A leitura atual mantém os mesmos achados qualitativos anteriores.
Foram preservadas cópias das saídas atuais junto desta nova auditoria.

## Questões antes limitadas

| ID | Tokens H / CE | Revisão das respostas completas |
| --- | ---: | --- |
| 5 | 1.309 / 1.512 | Ambas permanecem genéricas e inconsistentes com o procedimento esperado de ativação FISPACT-II/MCNP6.2. A continuação adiciona discussões de trapping e balanço que não suprem o núcleo da referência. CE apresenta até “decay from previous inventory” como parcela positiva de uma fórmula conceitual. Mais texto não corrige o problema. |
| 6 | 1.347 / 689 | H conclui 30–50 L/s, embora cite 97,5 L/s do loop; CE conclui 97,5 L/s. CE é preferível no resultado numérico, mas ainda contém fórmulas e comentários intermediários incorretos. |
| 13 | 1.359 / 1.480 | Ambas permanecem inadequadas. H pede esclarecimento e não reconstrói a prova específica. CE afirma incorretamente que as aproximações se restringem automaticamente a qualquer subespaço fechado, derivando hereditariedade de lushness. Não apresenta o argumento de densidade/repetição/epsilon arbitrário da referência. A resposta mais assertiva do CE não é melhoria neste caso. |
| 23 | 1.179 / 1.111 | Ambas continuam confundindo pertencer ao framework com superioridade teórica geral. H repete que Krum não teria garantias e chama o framework de “gold standard”; CE faz associações imprecisas com FLTrust e outros métodos. A conclusão adicional não corrige o núcleo. |

Inspeção complementar dos prompts: a questão 13 já contém no contexto híbrido
um trecho sobre pares de um subconjunto denso, todos os n e epsilons arbitrariamente
pequenos, concluindo que Y é lush. Também contém repetições infinitas e epsilon
1/m. Logo, o erro não pode ser atribuído apenas à ausência desses elementos no
contexto; o modelo não os utilizou adequadamente. Há fórmulas não decodificadas
no texto extraído, o que continua sendo uma limitação de apresentação.

Na questão 5, a ocorrência de FISPACT-II localizada no prompt trata de ativação
de materiais EUROFER/SS316L; não há ocorrência literal de MCNP. Isso não prova
ausência de todo suporte equivalente, mas recomenda auditar cobertura específica
de evidência antes de atribuir a falha somente à geração ou à ordenação.

## Restante da comparação

Mantêm-se os sinais qualitativos favoráveis ao CE em 8 (final 2%, ainda ambíguo),
9 (escala de aproximadamente 600 mil m³ em vez de 6,3 milhões), 14 (avaliação
absoluta) e 19 (softmax sem a identidade falsa apresentada por H). Na 3, CE
termina em aproximadamente 0,0131/dia, embora a conta acessória em segundos
continue errada. Nas duas ordens persistem os erros das questões 4, 7 e 12.
As respostas de 15, 17, 24 e 25 continuam atendendo ao núcleo esperado, com
ressalvas acessórias já descritas na revisão anterior.

Esta é uma avaliação qualitativa do assistente, sem juiz oficial e sem novos
percentuais de acurácia. Não há base para transformar a lista de melhorias
parciais em número de novos acertos. A questão 13 é um exemplo de risco
específico da ordem CE: uma resposta completa e confiante também pode estar errada.

## Decisão sugerida

Manter a seleção híbrida com ordem CE como próxima candidata experimental,
preservando prompt e geração de 2.048 tokens, agora com evidência local menos
afetada por corte de saída. Antes do ZIP, integrar a variante ao `answer()` e
validar a preservação do conjunto no contexto real e o fallback. Não misturar
uma mudança adicional de prompt, chunking ou parâmetros na mesma submissão.

O híbrido de 60% segue sendo o melhor resultado oficial informado. Esta rodada
não estabelece que a candidata o superará e não mede tempo total na H100.
Nenhum pacote ou código de inferência foi alterado nesta revisão.

## Artefatos

[Auditoria e hashes](../../experiments/results/colab-selection-order-review-2048-2026-11-14/audit.json),
[comparação com o registro de 1.024 tokens](../../experiments/results/colab-selection-order-review-2048-2026-11-14/comparison-with-1024.json).
As cópias `hybrid-answers-2048.json` e `ce100-answers-2048.json` estão na mesma pasta.
`tools/audit_colab_results.py` passou a aceitar `--hybrid-results` e `--ce-results`
para identificar arquivos explicitamente, sem depender do nome atribuído ao download.
