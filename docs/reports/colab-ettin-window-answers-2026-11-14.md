# Respostas Colab: Ettin 512 e 1.024

## Conclusão

A ampliação da janela alterou sete respostas, com melhora clara na estimativa
numérica da questão 9. Não houve evidência qualitativa de melhoria generalizada.
A questão 13 deixa de ser cortada, mas continua sem a justificativa específica
esperada. A resposta à questão 19 contém uma fórmula falsa nas duas janelas,
enquanto o MiniLM corrigido havia apresentado a fórmula correta sem essa
contradição. Esses resultados não sustentam substituir o MiniLM por Ettin
apenas pelo proxy de ordenação mais alto.

Revisão qualitativa do assistente frente ao `sample_qa.json`, sem juiz oficial,
sem nova inferência e sem converter julgamentos parciais em acurácia binária.

## Integridade da comparação

Foi criada `tools/audit_reranker_answer_comparison.py`, que verifica quatro
variantes: ordem híbrida, MiniLM corrigido, Ettin 512 e Ettin 1.024. Todas usam
a **mesma seleção híbrida** e os mesmos textos de chunks. Os metadados de
geração registrados coincidem, exceto pelo hash esperado de cada entrada.
Todos usam OLMo 3 7B Instruct, NF4/FP16, greedy e até 2.048 tokens gerados.

Foram conferidos IDs, referência oficial por questão, SHA-256 dos arquivos de
entrada, sistema do `main.py`, pergunta e mensagem user montada com o contexto
congelado. As saídas têm exatamente as 22 questões ativas, sem respostas vazias,
e status `ok` em todos os casos. As respostas não são as da LLM 32B oficial.

Entre Ettin 512 e 1.024:

- Oito prompts mudam: 7, 9, 11, 13, 14, 23, 24 e 25.
- Sete respostas mudam: 9, 11, 13, 14, 23, 24 e 25.
- As 14 perguntas com prompts idênticos produzem respostas idênticas byte a byte.
- A pergunta 7, embora reordenada, também produz a mesma resposta.
- Portanto **15 das 22 respostas são idênticas** entre as janelas.

Isso é consistente com uma ablação controlada de reordenação, sem evidência de
alteração de prompts ou configurações não declarada nos arquivos. Permanecem
ausentes revisão exata de pesos/tokenizer, GPU, device map, duração e motivo
de término. Os metadados iguais não permitem provar igualdade desses itens
não registrados. As cópias híbrida e MiniLM vêm dos snapshots da rodada de
2.048 tokens anteriormente auditada.

| Medida | Híbrido | MiniLM | Ettin 512 | Ettin 1.024 |
| --- | ---: | ---: | ---: | ---: |
| Respostas concluídas | 22 | 22 | 22 | 22 |
| IDs que atingiram o teto de saída | — | — | 13 | — |
| Média de tokens gerados | 647,05 | 633,95 | 704,32 | 716,95 |
| Maior pico alocado registrado, GiB | 10,82 | 10,82 | 10,82 | 10,82 |
| Maior pico reservado registrado, GiB | 12,18 | 12,18 | 12,17 | 12,17 |

O status `ok` não significa resposta correta nem término natural. No Ettin
512, a questão 13 usa todos os 2.048 tokens e termina no meio de uma referência;
há evidência textual de corte. Com 1.024 tokens **de entrada no reranker**, a
resposta da mesma questão usa 1.790 tokens **de saída na LLM** e termina completa.
São limites de componentes diferentes. Não houve OOM registrado; o custo de
geração em tempo não pode ser comparado, pois a duração não foi salva.

## Sete respostas alteradas

| ID | Ettin 512 → 1.024 | Avaliação |
| --- | --- | --- |
| 9 | Cerca de 6,5 milhões de m³ → cerca de 411 mil m³ acima do solo | Melhora numérica clara: a referência permite 410.589 m³ para a parte acima do solo. Ainda há confusão entre volume e área e hesitação excessiva. O MiniLM já estimava aproximadamente 600 mil m³ para o volume total, próximo dos 615.883,5 m³ da referência. |
| 11 | Muda a explicação genérica de hierarquia/codebooks | Continua parcial. Não explicita adequadamente codebooks maiores nas primeiras camadas e balanced k-means seletivo na última. A hierarquia é apresentada como diferencial de modo simplificado. Não há ganho claro de suporte factual. |
| 13 | Resposta longa/cortada que afirma hereditariedade → resposta completa que questiona a transferência para subespaços | A nova resposta evita parte da afirmação indevida, mas reformula a pergunta como sendo sobre um subespaço arbitrário e não justifica o Y construído. Não usa corretamente densidade, repetições infinitas e epsilon arbitrário, que já estavam no contexto. Ambas continuam inadequadas à referência. |
| 14 | Reescrita e expansão da explicação de métricas [0,1] | Ambas atendem ao núcleo de interpretabilidade absoluta e extremos significativos; não há ganho claro de correção. A 1.024 adiciona uma explicação imprecisa sobre saturação. |
| 23 | Reescreve a distinção entre frameworks | Ambas sugerem que alternativas seriam em geral menos rigorosas/heurísticas. Não esclarecem adequadamente que pertencer à família não implica superioridade de desempenho. Sem ganho claro. |
| 24 | Resposta de 486 → 785 tokens | Ambas incluem autodiagnóstico, incerteza e solicitação de supervisão/feedback. A 1.024 é mais extensa, sem mudança clara do núcleo correto. “Self-supervision” continua sendo terminologia imprecisa nesse contexto. |
| 25 | Resposta de 618 → 499 tokens | Ambas explicam kernel/dinâmica fixos e espaçamento variável. A 1.024 evita a afirmação acessória sobre todas as matrizes A/B/C de Mamba dependerem da entrada, mas mantém generalização sobre RNNs. Núcleo atendido nas duas. |

Reduzir truncamento de 101 para 23 pares não se traduziu em 78 evidências úteis
novas: na ablação avaliada pela LLM, o conjunto de chunks é o mesmo. O CE passa
a enxergar mais de algumas passagens durante a pontuação e muda sua ordem.
É esse efeito indireto que pode influenciar a resposta. A pergunta 9 mostra
uma mudança que o proxy documental praticamente não antecipava.

## Quinze respostas idênticas

| ID | Avaliação comum às duas janelas Ettin |
| --- | --- |
| 1 | Captura excitação interna/radiativa versus interface, mas torna a criação de pares elétron-lacuna excessivamente universal. |
| 2 | Captura perturbação do feixe, com explicação excessiva e exemplos acessórios; núcleo atendido. |
| 3 | Usa 53 dias e termina em aproximadamente 0,013/dia. A conta acessória em SI dá 1,47×10^-7 em vez de aproximadamente 1,514×10^-7 s^-1; há também imprecisão sobre algarismos significativos. |
| 4 | Conclui cerca de uma semana, contrariando a ordem de 10^7 segundos da referência. |
| 5 | Explicação genérica de produção/decay/traps, sem o procedimento FISPACT-II esperado. Não resolve a referência específica. |
| 6 | Conclui 97,5 L/s, próximo dos 98 L/s esperados. Introduz equivalências erradas: 97,5 L/s não são 97.500 L/h, nem 97,5 kg/s com densidade de 0,53 kg/L. |
| 7 | Aceita a premissa on-line; a referência exige corrigir que o nitrogen hot trap é off-line. Dar apenas 500–600 °C é insuficiente. |
| 8 | Conclui 2%, mas mistura versões/dados incompatíveis de fluxo da ICS. Resultado principal alinhado, com contradições intermediárias. |
| 10 | Discute zonas de manutenção sem apresentar adequadamente os dois limiares da referência. |
| 12 | Aceita comparação injusta e recomenda ablação sem imutabilidade, contrariando a referência sobre baselines que já têm essas restrições. |
| 15 | Calcula corretamente o intervalo [128,170]. |
| 17 | Relaciona pesos, espaço alcançável e representatividade; atende ao núcleo, mas exagera ao falar em espaço viável “exato”. |
| 19 | Nomeia softmax e inclui a fórmula exponencial correta, mas antes afirma falsamente que w_i/sum(w) é equivalente. A pergunta pede justamente a definição matemática, tornando essa contradição material. |
| 20 | Resposta genérica sobre gerar/refinar representações, sem explicar adequadamente o papel de R no teste ao não dispor das representações SC reais das classes não vistas. |
| 22 | Destaca composição/diversidade de augmentations, mas não explica bem a tarefa contrastiva trivial mesmo com pares corretamente identificados. |

## Comparação com MiniLM e decisão

Os ganhos antes observados de MiniLM sobre a ordem híbrida nas questões 6 e 8
também aparecem, com ressalvas, no Ettin. Isso não estabelece ganho adicional
do Ettin sobre MiniLM. Na questão 9, Ettin 512 regride em relação ao MiniLM;
Ettin 1.024 recupera uma estimativa aceitável para a parte acima do solo.
Na questão 19, o MiniLM apresenta a fórmula correta sem a equivalência falsa
introduzida nas duas variantes Ettin. Erros de 4, 7, 12 e 13 permanecem em
todas as variantes auditadas.

Portanto, os dados disponíveis **não mostram superioridade convincente do
Ettin sobre o MiniLM na geração**. Também não demonstram superioridade global
da janela 1.024 sobre 512: há uma melhora clara na 9, mudanças qualitativas
em outras questões e maior custo de reranking medido na execução local.
Não foi calculada acurácia oficial nem significância estatística.

Manter o híbrido de 60% como melhor referência oficial informada. Para a próxima
submissão experimental, os novos resultados não oferecem motivo forte para
abandonar a candidata seleção híbrida + ordem MiniLM corrigido. Se o objetivo
for continuar especificamente a investigação Ettin, a janela 1.024 é uma
candidata plausível para um teste de QA oficial, com custo maior e sem garantia
de ganho. Não misturar alterações de prompt na mesma submissão ao tentar
atribuir o resultado à reordenação.

Os erros acessórios e a verbosidade merecem uma ablação futura de instruções
mais diretas, sempre comparando o mesmo prompt entre os rerankers. Não deve
ser incluída silenciosamente nesta comparação já realizada.

## Reprodução e artefatos

```bash
python3 tools/audit_reranker_answer_comparison.py \
  --output-dir /tmp/ettin-answer-review
```

[Auditoria estruturada](../../experiments/results/colab-ettin-window-answer-review-2026-11-14/audit.json)
inclui hashes, medidas por variante e comparação textual por questão. Cópias das
quatro saídas foram preservadas na mesma pasta para evitar perda por substituição
posterior dos arquivos recebidos. Não houve alteração do notebook, geração
adicional, novas dependências, novos pesos ou mudança no pipeline de submissão.
O script de auditoria foi executado e os arquivos passaram em verificações de
integridade; a revisão semântica desta página é manual pelo assistente.
