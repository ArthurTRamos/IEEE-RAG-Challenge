# Análise da regressão do reranker top 100

O resultado informado caiu de **60% no híbrido para 55% no CE top 100**.
O usuário confirmou que não recebeu respostas, acertos por pergunta, resultados
por domínio ou tempo da submissão. Portanto esta análise identifica mecanismos
plausíveis nas 22 perguntas públicas, sem atribuir causalmente os cinco pontos
percentuais nem assumir que a mesma diferença ocorreu nessa amostra.

## Comparabilidade dos pacotes

A comparação dos ZIPs de 10 e 14/11 encontrou alterações somente em `main.py`
e `retrieval.py` entre os arquivos compartilhados. O segundo acrescenta
`reranking.py` e os arquivos do cross-encoder. Modelos de embedding e Docling
permanecem idênticos. O diff dos dois módulos mantém BM25, RRF, prompt, orçamento
de 10.000 tokens de texto e geração determinística com até 2.048 tokens.
As diferenças habilitam o CE sobre os 100 primeiros chunks, antes da seleção
do contexto. Registro: [archive-comparison.json](../../experiments/results/public22-ce-regression-audit-2026-11-14/archive-comparison.json).

O ranking híbrido define os candidatos; **dentro dos 100, a ordem passa a ser
inteiramente determinada pelo CE**, salvo empates. Não há combinação entre a
confiança do híbrido e a do CE. A cauda permanece depois dos 100.

## Evidência nova: a pergunta também pode ser truncada

A implementação instalada do Sentence Transformers usa `longest_first` na
tokenização de pares. Essa política remove tokens da sequência mais longa,
podendo atingir tanto o chunk quanto a pergunta, conforme a
[documentação de Transformers](https://huggingface.co/docs/transformers/main/en/pad_truncation).
Reproduzimos a tokenização com o tokenizer local do modelo, sem nova inferência.

Na **questão 13**, a pergunta matemática tem 308 tokens sem marcadores especiais.
Com o limite conjunto de 512, **64 dos 100 candidatos** são pontuados com uma
pergunta encurtada. **26 dos 45 chunks enviados à LLM** vieram desses pares.
Nenhuma outra pergunta desta amostra sofreu esse truncamento da consulta.

Exemplos: para `AI-007.pdf#8`, restam 285 tokens da pergunta; para
`AI-007.pdf#16`, 255. Em ambos, desaparece o final que pergunta por que Y é lush.
O CE pode estar pontuando a afinidade com a formulação do teorema sem enxergar
a solicitação final da questão. O prompt da LLM recebe a pergunta original:
o defeito observado está na informação disponível **durante o reranking**.

Há 144/2.200 pares acima de 512 tokens no top 100. Entre os chunks efetivamente
selecionados, 61/1.091 foram pontuados com pares truncados. Esses chunks ocupam
34.475/217.452 tokens de texto dos contextos (15,85%). Isso mede o espaço
ocupado pelos chunks afetados, **não** a quantidade de tokens descartados.
Uma parcela substancial desse fenômeno se concentra na pergunta 13; não é
evidência de que todas as perguntas sofram igualmente.

## O CE muda bastante a seleção e a posição das evidências

Recontagem dos contextos reais, com o tokenizer OLMo local e validação da regra
de parar no primeiro overflow:

| Medida | Híbrido | CE 50 | CE 100 | CE 200 |
| --- | ---: | ---: | ---: | ---: |
| Média de chunks no contexto | 52,82 | 52,27 | 49,59 | 49,18 |
| Retenção média do conjunto original | 100% | 95,60% | 69,39% | 61,71% |
| Ocorrências retiradas / acrescentadas | 0 / 0 | 37 / 25 | 353 / 282 | 446 / 366 |
| Chunks do top 10 original ainda no contexto, média | 10 | 9,91 | 9,18 | 8,82 |
| Interseção entre os dois top 10, média | 10 | 4,91 | 4,59 | 4,50 |
| Tokens de texto efetivamente usados, média | 9.806 | 9.883 | 9.884 | 9.841 |
| Documentos distintos no contexto, média | 9,45 | 9,45 | 8,82 | 9,18 |

No CE 100, 18 ocorrências do top 10 original deixam completamente o contexto.
Dos 353 chunks retirados, 120 são de documentos grau 2; entram 105 desse grau.
Mesmo preservando o recall documental agregado, podem mudar os fatos presentes.
O número menor de chunks não prova piora: os selecionados são mais longos,
e o orçamento usado **aumentou**, em vez de diminuir. Assim, orçamento ocioso
por overflow não é a explicação principal sugerida por esses dados.

A alteração também afeta posição: na **questão 24**, `AI-021.pdf#33`, que
descreve autodiagnóstico, confiança e solicitação de feedback por aprendizagem
ativa, cai de **1 para 16**; `AI-021.pdf#55`, que desenvolve o mesmo mecanismo,
cai de **4 para 18**. Ambos permanecem no contexto de 40 chunks. O caso mostra
perda de prioridade de evidência diretamente relacionada à resposta, sem
demonstrar que a LLM a ignorou. A literatura documenta sensibilidade à posição
em contextos longos, mas não valida esse efeito especificamente para esta
execução de OLMo: [Lost in the Middle](https://arxiv.org/abs/2307.03172).

Na **questão 7**, `DONES-035.pdf#63` (nitrogen trap off-line) passa de 12 para
57, e `DONES-035.pdf#72` (operação/getter a 600 °C) de 15 para 62. Ambos saem
do contexto, cujo corte passa de 52 para 42 chunks. Porém `DONES-035.pdf#64`
e `DONES-029.pdf#40` ainda mencionam operação off-line, agora nas posições 20
e 23. Não seria correto afirmar que toda a evidência para corrigir a premissa
da pergunta desapareceu. Esses são casos para auditoria, não erros de QA medidos.

## Por que as métricas anteriores não anteciparam a queda

O nDCG proxy@50 aumentou de 0,6156 para 0,6332 e a média de chunks de documentos
grau 2 de 26,73 para 27,27. Trata-se de ganho pequeno, com piora do proxy em oito
das 22 perguntas. O nDCG documental@10 já havia caído de 0,8816 para 0,8793.

Os graus vêm de julgamentos de IA **por documento**. Uma capa, referência ou
explicação lateral do artigo correto recebe o mesmo grau de uma passagem que
responde à pergunta. Na questão 24, por exemplo, perder 17 chunks grau 2 não
equivale a perder 17 evidências necessárias: vários trechos retirados discutem
definições gerais e governança, enquanto evidências centrais permanecem.

Outras observações restringem as hipóteses:

- Capa/metadados candidatos + bibliografia sobem de 53/1.162 para 66/1.091
  ocorrências no contexto (4,56% para 6,05%). Há mais ruído estrutural estimado,
  mas a classificação é heurística e não prova inutilidade factual.
- Mesmo domínio no top 50 já era 99,64% e passa a 99,73%. Um filtro rígido
  IA/DONES tem pouco espaço de ganho demonstrado nesta amostra.
- Em DONES, o CE 100 retira 57 chunks grau 2 e acrescenta 37; em IA, retira
  63 e acrescenta 68. É uma assimetria local que merece acompanhamento, sem
  permitir inferir os resultados oficiais por domínio.
- O modelo foi treinado em MS MARCO Passage Ranking, conforme seu
  [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
  Isso não garante discriminação de demonstrações matemáticas, tabelas,
  ressalvas ou versões de instalações científicas. Inadequação ao domínio é
  uma hipótese, não um diagnóstico demonstrado pelo treinamento de origem.

## Melhorias e testes prioritários

1. **Preservar a pergunta inteira no CE.** Para perguntas que cabem, usar
   truncamento somente da segunda sequência e reservar o espaço da pergunta
   e dos tokens especiais. Para perguntas maiores que a janela, definir
   fallback explícito ao híbrido; não cortar silenciosamente a solicitação.
   Começar com `ce100-query-preserved` e comparar os pares afetados. Isso não
   resolve o truncamento do chunk; pode até reduzir o espaço disponível a ele.
   Para passagens longas, uma ablação posterior pode usar janelas com overlap
   e agregar scores, mantendo o chunk original na seleção. Essa agregação
   precisa ser validada e aumenta o número de pares. Não basta elevar
   `max_length`: o modelo local possui limite de posições de 512.

2. **Separar seleção e ordem.** Após corrigir a consulta, comparar: conjunto e
   ordem híbridos; conjunto híbrido em ordem CE; conjunto CE em ordem híbrida;
   conjunto e ordem CE. No teste de ordem pura, selecionar primeiro o contexto
   híbrido e apenas permutar esse conjunto, sem refazer o packing. CE top 50
   não isola perfeitamente ordem: houve 37 saídas e 25 entradas nos contextos
   reais, embora o conjunto do top 50 fixo seja idêntico. Essa ablação permite
   atribuição melhor em um teste futuro com a LLM.

3. **Dar ao CE influência parcial.** Testar fusão entre posições híbrida e CE
   no mesmo pool de 100: `(1-alpha)/(60+r_h) + alpha/(60+r_ce)`, com `alpha`
   inicialmente 0,25 e 0,5; preservar a cauda. Usar nomes como
   `hybrid-ce100-query-preserved-rankblend-a0p25`. Não somar logits CE aos
   scores RRF brutos. Essa é uma hipótese de menor perturbação, não garantia
   de preservar evidências; medir novamente as saídas e a cobertura factual.

4. **Avaliar suporte da resposta por passagem.** Auditar a união dos trechos
   que entraram/saíram e dos primeiros resultados, marcando suporte direto,
   parcial, ausência de suporte e contradição. Incluir valores/unidades,
   condições de teoremas, negações e versões. Medir cobertura dos fatos
   necessários no contexto e a posição da primeira evidência, além do proxy
   documental. Respostas públicas servem à avaliação offline, nunca à
   recuperação em produção. Evitar escolher dezenas de configurações pelo
   mesmo conjunto de 22 perguntas sem validação separada.

Penalização leve de bibliografia, redução de redundância e outro reranker
ficam para uma segunda rodada, uma variável por vez. Não remover capas ou
referências indiscriminadamente, impor filtro de domínio ou aumentar para
200 candidatos como primeira resposta à regressão. CE 200 já perdeu recall
documental grau 2 no contexto (97,73% para 94,70%) na avaliação local.

O híbrido de 60% continua sendo a referência de submissão. Não há resultados
de QA para as melhorias propostas. A correção de truncamento é a primeira
intervenção concreta recomendada; a próxima decisão de ranking deve usar
evidência por passagem, além dos proxies atuais.

## Recursos e reprodução

A análise não altera o pipeline ou os ZIPs e não executa geração. O pacote CE
medido tem 482.431.786 bytes (482,4 MB), dentro de 1 GB. Preservar a consulta,
reordenar conjuntos e combinar posições dispensam novos pesos/dependências e
podem reutilizar os mesmos scores em cada ablação compatível. Corrigir a
tokenização exige recalcular scores dos pares afetados. Janelas múltiplas e
outros modelos precisam de nova medição de tempo, memória e tamanho.

O tempo oficial do CE não foi informado. Os 184,98 s locais para 4.400 pares
em CPU com quatro threads excluem parsing, embeddings e geração; não comprovam
o limite total de 60 minutos na H100. Esta auditoria não mede nova latência CE.

```bash
.venv/bin/python tools/analyze_reranking_regression.py \
  --output-dir experiments/results/public22-ce-regression-audit-NOVA-RUN
```

Resultados: [resumo](../../experiments/results/public22-ce-regression-audit-2026-11-14/summary.json),
[por pergunta](../../experiments/results/public22-ce-regression-audit-2026-11-14/per-question.json)
e [manifesto com hashes](../../experiments/results/public22-ce-regression-audit-2026-11-14/manifest.json).
A ferramenta confere IDs, catálogo, textos dos chunks, prefixo selecionado
e orçamento/primeiro overflow nas quatro variantes. Os exemplos de passagens
acima foram inspecionados no catálogo e nos rankings congelados, sem criar
novos julgamentos automáticos de suporte factual.
