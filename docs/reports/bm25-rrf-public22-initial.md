# Comparação inicial: denso, BM25 e RRF

Execução local em 10/11/2026, sem carregar o LLM gerador. Os três métodos foram
avaliados nas 22 perguntas ativas e nos mesmos 5.174 chunks de 64 PDFs. Os IDs
16, 18 e 21 não participam da avaliação.

## Configuração e reprodução

- Fonte congelada: `experiments/runs/baseline-public22-ranking/`.
- Artefatos: `experiments/runs/bm25-rrf-public22-initial/`, incluindo
  `summary.json`, `manifest.json` e rankings/contextos/métricas por método.
- Modelo denso: MiniLM do baseline; scores reutilizados da execução fonte.
- BM25: `rank-bm25==0.2.2`, `k1=1.25`, `b=0.75`, demais padrões da biblioteca.
- Tokenização lexical: palavras/números Unicode (`\w+`), `casefold`, sem
  stemming e sem remoção de stopwords, tanto para chunks como para perguntas.
- RRF: constante 60, posições a partir de 1, pesos iguais para os dois canais.
  Todos os chunks recebem voto denso; somente scores BM25 positivos recebem
  voto lexical. Não há corte fixo de candidatos. Empates BM25/RRF seguem a
  ordem original dos chunks.
- Contexto: orçamento original de 10.000 tokens de texto dos chunks. Para no
  primeiro overflow e mantém o fallback original por caracteres para um
  primeiro chunk maior que o orçamento. Não há corte em dez chunks.

Foi baixado apenas o tokenizer do
[OLMo oficial](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct), revisão
`ac0587e4a7744a551c059d8cd17ba220bc940dae`, em `models/olmo-tokenizer/`.
Os arquivos somam aproximadamente 9,68 MB, incluindo metadados do download.
O tokenizer foi carregado localmente, com template de chat disponível e 100.278
tokens no vocabulário. Os pesos de 32B não foram baixados.

```bash
.venv/bin/python tools/download_tokenizer.py \
  --revision ac0587e4a7744a551c059d8cd17ba220bc940dae

.venv/bin/python tools/compare_retrieval.py \
  --source-run experiments/runs/baseline-public22-ranking \
  --tokenizer models/olmo-tokenizer \
  --k1 1.25 --b 0.75 --rrf-k 60 \
  --output-dir experiments/runs/bm25-rrf-public22-initial
```

O diretório de saída precisa estar vazio; para repetir a execução existente,
escolha um novo nome. O replay conferiu hashes dos PDFs, julgamentos, perguntas
e conteúdo de todos os chunks e reproduziu exatamente a ordem densa das 22
perguntas. As médias documentais do ranking denso coincidem integralmente com
as da execução original do usuário. Nenhum parsing ou embedding foi repetido.

## Resultados

Médias sobre as 22 perguntas. Recall de contexto considera os documentos
distintos presentes em **todos** os chunks selecionados, sem corte em dez.

| Método | nDCG documental@10 | Recall grau 2 no contexto completo | Recall graus 1–2 no contexto completo | Média de chunks | Média de tokens dos chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Denso | 0,8458 | 0,9773 | 0,9499 | 55,77 | 9.853,91 |
| BM25 | 0,8044 | 0,9659 | 0,9662 | 47,59 | 9.865,77 |
| Híbrido RRF | 0,8741 | 0,9773 | 0,9784 | 51,86 | 9.897,09 |

Os contextos contêm de 42 a 67 chunks no denso, de 28 a 58 no BM25 e de 31 a
64 no híbrido. Todos ficaram dentro de 10.000 tokens de texto dos chunks.
Cabeçalhos, pergunta e template de chat ficam fora desse orçamento, como no
baseline original; ele não representa a janela máxima de contexto do modelo.

O híbrido melhorou nDCG documental@10 em aproximadamente 0,0282 em relação ao
denso, com o mesmo recall de grau 2 no contexto completo. Isso demonstra melhor
ordenação documental neste conjunto, sem evidência de ganho em acurácia de QA.

No contexto completo:

- Denso e híbrido: o único documento de grau 2 ausente foi `AI-006` na pergunta
  12. A pergunta ainda recebeu outro documento de grau 2.
- BM25: além de `AI-006` na pergunta 12, faltou `DONES-018` na pergunta 2.
- Recall grau 2 médio por domínio: denso/híbrido com 1,0000 em DONES e 0,9583
  em IA; BM25 com 0,9750 em DONES e 0,9583 em IA. O agregado da tabela é a
  média das 22 perguntas, não uma simulação do score oficial balanceado.

## Recursos e limites da comparação

Construção do BM25: aproximadamente 0,337 s. Replay completo, incluindo
validação, carregamento do tokenizer e serialização dos artefatos: 27,96 s.
Pico de RSS do processo: aproximadamente 2,09 GiB, incluindo os grandes JSONs
de ranking em memória; exclui processos filhos e memória de GPU.

Esses tempos não incluem os 350,05 s de construção do índice registrados na
execução fonte. Também não incluem carregamento ou geração do LLM. Portanto,
não validam a restrição de 60 minutos na H100 nem devem ser comparados como
tempos totais de execução das três arquiteturas.

BM25 e RRF não exigem novos pesos e usam uma dependência já instalada no
ambiente oficial. Os JSONs locais e o tokenizer de desenvolvimento ficam fora
do ZIP de submissão. O código de inferência continua usando o tokenizer do
modelo disponibilizado pela avaliação. O impacto adicional no pacote é somente
código; o tamanho final do ZIP não foi reconstruído nesta execução.

Os julgamentos são estimativas de IA no nível de documento, com negativos
provisórios. Grau 2 exige suporte a pelo menos parte da resposta; a métrica não
confirma que o chunk selecionado contém esse suporte. Acurácia final e cobertura
dos fatos permanecem sem medição.

## Validação de implementação

Passaram 20 testes de contrato, fórmulas BM25/RRF, orçamento de contexto,
equivalência com o starter, geração de artefatos e validação de replay. O padrão
de `src/main.py` permanece denso; as variantes podem ser selecionadas pela
constante `RETRIEVAL_METHOD`. Nenhuma submissão foi feita.
